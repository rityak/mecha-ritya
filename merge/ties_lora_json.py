import torch
import logging
from typing import Dict, Any
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .ties_json import trim_task_vector, elect_sign, get_model_stock_t, geometric_median
from .lib.parse_utils import parse_alphas, parse_json_params
from .lib.tensor_utils import fix_non_finite_tensors, validate_result

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def disjoint_merge(
    task_vectors: Tensor,
    elected_signs: Tensor,
    weights: Tensor,
    apply_stock: bool = False,
    apply_median: bool = False,
    cos_eps: float = 1e-6,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
) -> Tensor:
    """
    Merge task vectors by keeping only values whose signs match the elected sign.
    Optionally apply Model Stock scaling or Geometric Median.
    
    According to TIES paper: τ^p_m = mean(τ^p_t | sign(τ^p_t) == γ^p_m)
    
    Args:
        task_vectors: Stacked trimmed task vectors, shape (num_models, ...)
        elected_signs: Elected sign vector, shape (...)
        weights: Weights for each model, shape (num_models,)
        apply_stock: If True, apply Model Stock scaling
        apply_median: If True, apply Geometric Median instead of weighted average
        cos_eps: Epsilon for cosine similarity (for Model Stock)
        eps: Minimum distance for Geometric Median
        maxiter: Maximum iterations for Geometric Median
        ftol: Tolerance for Geometric Median convergence
    
    Returns:
        Merged task vector
    """
    # Get signs of each task vector
    signs = torch.sign(task_vectors)
    
    # Create mask: True where sign matches elected sign
    # Expand elected_signs to match task_vectors shape
    elected_signs_expanded = elected_signs.unsqueeze(0)
    mask = (signs == elected_signs_expanded)
    
    # Apply mask to task vectors (filtered_delta)
    filtered_vectors = task_vectors * mask
    
    # Count how many models contribute to each parameter
    weights_expanded = weights.view(-1, *([1] * (task_vectors.dim() - 1)))
    param_counts = (weights_expanded * mask).sum(dim=0)
    
    if apply_median:
        # Apply Geometric Median to filtered vectors
        # Convert filtered_vectors to list of tensors (one per model)
        filtered_list = [filtered_vectors[i] for i in range(filtered_vectors.shape[0])]
        merged = geometric_median(*filtered_list, eps=eps, maxiter=maxiter, ftol=ftol)
    else:
        # Standard weighted average
        weighted_sum = (filtered_vectors * weights_expanded).sum(dim=0)
        
        # Avoid division by zero
        param_counts = torch.where(param_counts == 0, torch.ones_like(param_counts), param_counts)
        merged = weighted_sum / param_counts
        
        # Apply Model Stock scaling if requested
        if apply_stock:
            # Convert filtered_vectors to list for get_model_stock_t
            filtered_list = [filtered_vectors[i] for i in range(filtered_vectors.shape[0])]
            t = get_model_stock_t(filtered_list, cos_eps=cos_eps)
            # Expand t to match merged shape if needed
            if t.dim() < merged.dim():
                t = t.view(*t.shape, *([1] * (merged.dim() - t.dim())))
            merged = merged * t
    
    return merged


@merge_method(identifier="🔨_ties_lora_with_json", register=True)
def ties_lora_merging_with_json(
    *lora_delta_dicts: Parameter(Tensor),
    json_params: Parameter(str),
    **kwargs
) -> Return(Tensor):
    """
    Merges several LoRA delta tensors using the TIES-MERGING algorithm (TRIM, ELECT SIGN & MERGE),
    with global settings only (no block-specific configs).
    
    TIES-MERGING addresses interference when merging LoRA deltas by:
    1. Trimming redundant parameters (keeping only top-k% by magnitude)
    2. Electing signs to resolve sign conflicts
    3. Disjoint merging (only merging values with matching signs)
    
    This method works with pre-computed LoRA deltas (already unfolded and scaled).
    It is called for each key separately by sd_mecha, which collects the results into a dictionary.
    
    Parameters:
    - *lora_delta_dicts: Variable number of LoRA delta tensors (one per LoRA)
    - json_params: JSON string containing global configuration
      Example: {
        "lambda": 1.0,
        "weights": "1.0,1.0,1.0",
        "density": 0.25,
        "vote_sgn": false,
        "apply_stock": false,
        "apply_median": false,
        "cos_eps": 1e-6,
        "eps": 1e-6,
        "maxiter": 100,
        "ftol": 1e-20
      }
    
    JSON Parameters (all global, applied to all keys):
    - lambda: Scaling parameter for final result (default: 1.0)
    - weights: Comma-separated weights for each LoRA (default: "1.0" for all)
    - density: Global density for TRIM step, 0.0-1.0 (default: 0.25)
    - vote_sgn: If True, use sum of vectors for sign election; if False, use sum of signs (default: False)
    - apply_stock: If True, apply Model Stock scaling (default: False)
    - apply_median: If True, use Geometric Median instead of weighted average (default: False)
    - cos_eps: Epsilon for cosine similarity in Model Stock (default: 1e-6)
    - eps: Minimum distance for Geometric Median (default: 1e-6)
    - maxiter: Maximum iterations for Geometric Median (default: 100)
    - ftol: Tolerance for Geometric Median convergence (default: 1e-20)
    
    Returns:
    Merged delta tensor for the current key
    """
    # Get the current key
    key = kwargs.get("key", "n/a")
    
    if len(lora_delta_dicts) == 0:
        return torch.tensor(0.0)
    
    # Parse JSON config
    params: Dict[str, Any] = parse_json_params(json_params)
    
    # Get global parameters
    lambda_val: float = params.get("lambda", 1.0)
    weights_str: str = params.get("weights", "1.0")
    density: float = params.get("density", 0.25)
    vote_sgn: bool = params.get("vote_sgn", False)
    apply_stock: bool = params.get("apply_stock", False)
    apply_median: bool = params.get("apply_median", False)
    cos_eps: float = params.get("cos_eps", 1e-6)
    eps: float = params.get("eps", 1e-6)
    maxiter: int = params.get("maxiter", 100)
    ftol: float = params.get("ftol", 1e-20)
    
    # Parse weights
    weight_values = parse_alphas(weights_str)
    
    # Validate weights count matches number of input tensors
    num_loras = len(lora_delta_dicts)
    if len(weight_values) != num_loras:
        raise ValueError(
            f"The number of weights ({len(weight_values)}) "
            f"does not match the number of LoRA tensors ({num_loras})."
        )
    
    # Validate density
    if density < 0.0 or density > 1.0:
        raise ValueError(f"Density must be between 0.0 and 1.0, got {density}")
    
    # Validate lambda
    if not isinstance(lambda_val, (int, float)):
        raise ValueError(f"lambda must be a number, got {type(lambda_val)}")
    
    # Validate vote_sgn
    if not isinstance(vote_sgn, bool):
        raise ValueError(f"vote_sgn must be a boolean, got {type(vote_sgn)}")
    
    # Validate apply_stock and apply_median
    if not isinstance(apply_stock, bool):
        raise ValueError(f"apply_stock must be a boolean, got {type(apply_stock)}")
    if not isinstance(apply_median, bool):
        raise ValueError(f"apply_median must be a boolean, got {type(apply_median)}")
    
    # Validate that apply_stock and apply_median are not both True
    if apply_stock and apply_median:
        raise ValueError("apply_stock and apply_median cannot both be True")
    
    # Validate global constants
    if cos_eps <= 0.0:
        raise ValueError(f"cos_eps must be positive, got {cos_eps}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if maxiter <= 0:
        raise ValueError(f"maxiter must be positive, got {maxiter}")
    if ftol <= 0.0:
        raise ValueError(f"ftol must be positive, got {ftol}")
    
    # Collect deltas (they are already tensors, one per LoRA for this key)
    deltas = []
    reference_device = None
    reference_dtype = None
    reference_shape = None
    
    for delta in lora_delta_dicts:
        # Convert to float32
        delta = delta.float()
        
        if reference_device is None:
            reference_device = delta.device
            reference_dtype = delta.dtype
            reference_shape = delta.shape
        
        deltas.append(delta)
    
    if not deltas:
        return torch.tensor(0.0)
    
    # Fix non-finite values
    deltas = list(fix_non_finite_tensors(tuple(deltas), key))
    
    # Step 1: TRIM - trim each delta with global density
    trimmed_deltas = []
    for delta in deltas:
        trimmed = trim_task_vector(delta, density)
        trimmed_deltas.append(trimmed)
    
    if not trimmed_deltas:
        return torch.zeros(reference_shape, dtype=reference_dtype, device=reference_device)
    
    # Stack deltas: shape (num_loras, ...)
    deltas_stacked = torch.stack(trimmed_deltas, dim=0)
    
    # Convert weights to tensor on same device/dtype as deltas
    weights_tensor = torch.tensor(
        weight_values,
        device=reference_device,
        dtype=reference_dtype
    )
    
    # Step 2: ELECT SIGN
    elected_signs = elect_sign(deltas_stacked, weights_tensor, vote_sgn=vote_sgn)
    
    # Step 3: DISJOINT MERGE
    merged_delta = disjoint_merge(
        deltas_stacked,
        elected_signs,
        weights_tensor,
        apply_stock=apply_stock,
        apply_median=apply_median,
        cos_eps=cos_eps,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol
    )
    
    # Step 4: Apply lambda scaling
    merged_delta = lambda_val * merged_delta
    
    # Final validation
    return validate_result(merged_delta, key)


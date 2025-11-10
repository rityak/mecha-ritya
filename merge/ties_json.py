import torch
import logging
from typing import Dict, Any, List, Tuple, Sequence, Optional
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .lib.parse_utils import parse_alphas, parse_json_params
from .lib.block_utils import get_user_block_name
from .lib.tensor_utils import fix_non_finite_tensors, validate_result
from .lib.weight_logger import WeightLogger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global weight logger instance
_weight_logger = WeightLogger()


def trim_task_vector(task_vector: Tensor, density: float) -> Tensor:
    """
    Trim task vector to keep only top-k% parameters by magnitude.
    
    Args:
        task_vector: Task vector to trim
        density: Fraction of parameters to keep (0.0-1.0, e.g., 0.2 for top-20%)
    
    Returns:
        Trimmed task vector with bottom (1-density)% values set to zero
    """
    if density <= 0.0:
        return torch.zeros_like(task_vector)
    if density >= 1.0:
        return task_vector
    
    # Compute absolute values
    abs_values = torch.abs(task_vector)
    
    # Calculate how many values to keep (top-k%)
    num_elements = task_vector.numel()
    num_to_keep = int(density * num_elements)
    
    if num_to_keep <= 0:
        return torch.zeros_like(task_vector)
    if num_to_keep >= num_elements:
        return task_vector
    
    # Find the threshold value (k-th largest value)
    # We need to find the (num_elements - num_to_keep + 1)-th smallest value
    k = num_elements - num_to_keep + 1
    threshold = torch.kthvalue(abs_values.flatten(), k).values
    
    # Create mask for values >= threshold
    mask = abs_values >= threshold
    
    return task_vector * mask


def get_model_stock_t(deltas: Sequence[Tensor], cos_eps: float) -> Tensor:
    """
    Compute Model Stock scaling coefficient based on cosine similarity between deltas.
    
    Approximate solution from mergekit: average of cos(theta).
    The expected value is 0, which accidentally corresponds with the implementation from the paper.
    This may be very unstable and the range is restricted to [-1, 1].
    
    Args:
        deltas: Sequence of delta tensors
        cos_eps: Epsilon for cosine similarity computation
    
    Returns:
        Scaling coefficient tensor
    """
    n = len(deltas)
    if n < 2:
        return torch.ones(1, device=deltas[0].device, dtype=deltas[0].dtype)
    
    cos = torch.nn.CosineSimilarity(dim=-1, eps=cos_eps)
    cos_thetas = [cos(deltas[i], deltas[i + 1]) for i in range(n - 1)]
    cos_theta = torch.stack(cos_thetas).mean(dim=0)
    
    t = (n * cos_theta / (1 + (n - 1) * cos_theta)).unsqueeze(-1)
    return t


def weighted_average(
    points: Sequence[Tensor],
    weights: Optional[Sequence[Tensor]] = None
) -> Tensor:
    """
    Compute weighted average of points.
    
    Args:
        points: Sequence of tensors (points)
        weights: Optional sequence of weights for each point
    
    Returns:
        Weighted average tensor
    """
    if weights is not None:
        weighted_sum = sum(p * weights[i] for i, p in enumerate(points))
        weight_sum = sum(weights)
        return weighted_sum / weight_sum
    else:
        return sum(points) / len(points)


def geometric_median_objective(median: Tensor, points: Tuple[Tensor, ...], weights: Tensor) -> float:
    """
    Compute geometric median objective function.
    
    Args:
        median: Current median estimate
        points: Tuple of point tensors
        weights: Weights for each point
    
    Returns:
        Objective value (float)
    """
    distances = torch.stack([torch.dist(point, median) for point in points])
    return torch.mean(distances * weights).item()


def geometric_median(
    *models: Tensor,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
) -> Tensor:
    """
    Compute Geometric Median using Weiszfeld algorithm.
    
    Source: https://github.com/krishnap25/geom_median/blob/main/src/geom_median/torch/weiszfeld_list_of_array.py
    
    Args:
        *models: Variable number of model tensors
        eps: Minimum distance to avoid division by zero
        maxiter: Maximum number of iterations
        ftol: Tolerance for convergence
    
    Returns:
        Geometric median tensor
    """
    if not models:
        raise ValueError("At least one model is required for geometric median")
    
    # Initialize with weighted average
    median = weighted_average(models)
    weights = torch.ones(len(models), device=models[0].device, dtype=models[0].dtype)
    new_weights = weights.clone()
    
    objective_value = geometric_median_objective(median, tuple(models), weights)
    
    # Weiszfeld iterations
    for _ in range(max(0, maxiter)):
        prev_obj_value = objective_value
        
        # Compute distances from median to each point
        denom = torch.stack([torch.dist(p, median) for p in models])
        new_weights = weights / torch.clamp(denom, min=eps)
        
        # Update median
        median = weighted_average(models, list(new_weights))
        
        # Check convergence
        objective_value = geometric_median_objective(median, tuple(models), weights)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            break
    
    return weighted_average(models, list(new_weights))


def elect_sign(
    task_vectors: Tensor,
    weights: Tensor,
    vote_sgn: bool = False
) -> Tensor:
    """
    Elect the sign for each parameter based on the weighted sum of task vectors or their signs.
    
    According to TIES paper: γ^p_m = sgn(Σ_t w_t * τ^p_t)
    
    Args:
        task_vectors: Stacked task vectors, shape (num_models, ...)
        weights: Weights for each model, shape (num_models,)
        vote_sgn: If True, use sum of vectors; if False, use sum of signs
    
    Returns:
        Elected sign vector with values +1, 0, or -1
    """
    # Expand weights to match task_vectors shape
    weights_expanded = weights.view(-1, *([1] * (task_vectors.dim() - 1)))
    weighted_task_vectors = task_vectors * weights_expanded
    
    if vote_sgn:
        # Sum weighted task vectors along the first dimension
        sum_vector = weighted_task_vectors.sum(dim=0)
        # Return sign of the sum
        return torch.sign(sum_vector)
    else:
        # Sum signs of weighted task vectors
        signs = torch.sign(weighted_task_vectors)
        sum_signs = signs.sum(dim=0)
        # Return sign of the sum of signs
        return torch.sign(sum_signs)


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


@merge_method(identifier="🔨_ties_merging_with_json", register=True)
def ties_merging_with_json(
        base_model: Parameter(Tensor),
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using the TIES-MERGING algorithm (TRIM, ELECT SIGN & MERGE),
    with the ability to set all parameters via a JSON string.
    
    TIES-MERGING addresses interference when merging models by:
    1. Trimming redundant parameters (keeping only top-k% by magnitude)
    2. Electing signs to resolve sign conflicts
    3. Disjoint merging (only merging values with matching signs)
    4. Optionally applying Model Stock scaling or Geometric Median
    
    Parameters:
    - base_model: Base model tensor (pre-trained model)
    - *tensors: Variable number of fine-tuned model tensors to merge
    - json_params: JSON string containing configuration
      Example: {
        "lambda": 1.0,
        "cos_eps": 1e-6,
        "eps": 1e-6,
        "maxiter": 100,
        "ftol": 1e-20,
        "vote_sgn": false,
        "apply_stock": false,
        "apply_median": false,
        "global": {
          "weights": "1.0, 1.0, 1.0",
          "densities": "0.2, 0.2, 0.2",
          "vote_sgn": false,
          "apply_stock": false,
          "apply_median": false
        },
        "IN": {
          "weights": "0.5, 0.3, 0.2",
          "densities": "0.2, 0.2, 0.2",
          "apply_stock": true
        }
      }
    
    JSON Parameters:
    Global constants (root level only):
    - cos_eps: Epsilon for cosine similarity in Model Stock (default: 1e-6)
    - eps: Minimum distance for Geometric Median (default: 1e-6)
    - maxiter: Maximum iterations for Geometric Median (default: 100)
    - ftol: Tolerance for Geometric Median convergence (default: 1e-20)
    
    Block parameters (can be in root, global, or per-block):
    - lambda: Scaling parameter (default: 1.0)
    - weights: Comma-separated weights for each model (default: "1.0" for all)
    - densities: Comma-separated densities for each model, 0.0-1.0 (default: "0.2" for all)
    - vote_sgn: If True, use sum of vectors for sign election; if False, use sum of signs (default: False)
    - apply_stock: If True, apply Model Stock scaling (default: False)
    - apply_median: If True, use Geometric Median instead of weighted average (default: False)
    """
    global _weight_logger
    
    if len(tensors) == 0:
        return base_model
    
    params: Dict[str, Any] = parse_json_params(json_params)
    
    # Get global lambda parameter
    lambda_val: float = params.get("lambda", 1.0)
    
    # Get global constants (only from root level)
    cos_eps: float = params.get("cos_eps", 1e-6)
    eps: float = params.get("eps", 1e-6)
    maxiter: int = params.get("maxiter", 100)
    ftol: float = params.get("ftol", 1e-20)
    
    # Get default global weights and densities
    global_config = params.get("global", {})
    if isinstance(global_config, str):
        # Legacy format: just weights string
        global_weights_str = global_config
        global_densities_str = "0.2"
        global_vote_sgn = False
        global_apply_stock = False
        global_apply_median = False
    else:
        global_weights_str = global_config.get("weights", "1.0")
        global_densities_str = global_config.get("densities", "0.2")
        global_vote_sgn = global_config.get("vote_sgn", params.get("vote_sgn", False))
        global_apply_stock = global_config.get("apply_stock", params.get("apply_stock", False))
        global_apply_median = global_config.get("apply_median", params.get("apply_median", False))
    
    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    
    # Get block-specific config or use global
    block_config = params.get(user_block_name, {})
    if isinstance(block_config, str):
        # Legacy format: just weights string
        weights_str = block_config
        densities_str = global_densities_str
        vote_sgn = global_vote_sgn
        apply_stock = global_apply_stock
        apply_median = global_apply_median
    else:
        weights_str = block_config.get("weights", global_weights_str)
        densities_str = block_config.get("densities", global_densities_str)
        vote_sgn = block_config.get("vote_sgn", global_vote_sgn)
        apply_stock = block_config.get("apply_stock", global_apply_stock)
        apply_median = block_config.get("apply_median", global_apply_median)
    
    # Log weights if they have changed
    features_str = ""
    if vote_sgn or apply_stock or apply_median:
        features = []
        if vote_sgn:
            features.append("vote_sgn")
        if apply_stock:
            features.append("apply_stock")
        if apply_median:
            features.append("apply_median")
        features_str = f", features=[{', '.join(features)}]"
    config_str = f"weights={weights_str}, densities={densities_str}, lambda={lambda_val}{features_str}"
    _weight_logger.log_weights(user_block_name, config_str, "ties_merging_with_json")
    
    # Parse weights and densities
    weight_values = parse_alphas(weights_str)
    density_values = parse_alphas(densities_str)
    
    # Validate lengths
    num_models = len(tensors)
    if len(weight_values) != num_models:
        raise ValueError(
            f"The number of weights for block '{user_block_name}' ({len(weight_values)}) "
            f"does not match the number of models ({num_models})."
        )
    if len(density_values) != num_models:
        raise ValueError(
            f"The number of densities for block '{user_block_name}' ({len(density_values)}) "
            f"does not match the number of models ({num_models})."
        )
    
    # Validate density values (should be between 0 and 1)
    for i, d in enumerate(density_values):
        if d < 0.0 or d > 1.0:
            raise ValueError(
                f"Density value {d} for model {i} in block '{user_block_name}' "
                f"must be between 0.0 and 1.0."
            )
    
    # Validate global constants
    if cos_eps <= 0.0:
        raise ValueError(f"cos_eps must be positive, got {cos_eps}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    if maxiter <= 0:
        raise ValueError(f"maxiter must be positive, got {maxiter}")
    if ftol <= 0.0:
        raise ValueError(f"ftol must be positive, got {ftol}")
    
    # Validate boolean parameters
    if not isinstance(vote_sgn, bool):
        raise ValueError(f"vote_sgn must be a boolean, got {type(vote_sgn)}")
    if not isinstance(apply_stock, bool):
        raise ValueError(f"apply_stock must be a boolean, got {type(apply_stock)}")
    if not isinstance(apply_median, bool):
        raise ValueError(f"apply_median must be a boolean, got {type(apply_median)}")
    
    # Convert to tensors
    weights_tensor = torch.tensor(weight_values, device=base_model.device, dtype=base_model.dtype)
    density_tensor = torch.tensor(density_values, device=base_model.device, dtype=base_model.dtype)
    
    # Fix non-finite tensors
    base_model = fix_non_finite_tensors((base_model,), key)[0]
    tensors = fix_non_finite_tensors(tensors, key)
    
    # Step 1: Compute task vectors and trim them
    task_vectors = []
    for i, tensor in enumerate(tensors):
        # Compute task vector: τ_t = θ_t - θ_init
        task_vector = tensor - base_model
        
        # Trim task vector: keep only top-k% by magnitude
        density = density_tensor[i].item()
        trimmed_task_vector = trim_task_vector(task_vector, density)
        
        task_vectors.append(trimmed_task_vector)
    
    if not task_vectors:
        return base_model
    
    # Stack task vectors: shape (num_models, ...)
    task_vectors_stacked = torch.stack(task_vectors, dim=0)
    
    # Step 2: Elect signs
    # Elect signs with vote_sgn option: γ^p_m = sgn(Σ_t w_t * τ^p_t) or sgn(Σ_t sgn(w_t * τ^p_t))
    elected_signs = elect_sign(task_vectors_stacked, weights_tensor, vote_sgn=vote_sgn)
    
    # Step 3: Disjoint merge
    # Merge only values whose signs match the elected sign
    # Optionally apply Model Stock scaling or Geometric Median
    merged_task_vector = disjoint_merge(
        task_vectors_stacked,
        elected_signs,
        weights_tensor,
        apply_stock=apply_stock,
        apply_median=apply_median,
        cos_eps=cos_eps,
        eps=eps,
        maxiter=maxiter,
        ftol=ftol
    )
    
    # Step 4: Compute final result: θ_m = θ_init + λ * τ_m
    result = base_model + lambda_val * merged_task_vector
    
    # Final validation
    return validate_result(result, key)


import torch
import logging
from typing import Dict, Any, List, Tuple
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


def elect_sign(weighted_task_vectors: Tensor) -> Tensor:
    """
    Elect the sign for each parameter based on the weighted sum of task vectors.
    
    According to TIES paper: γ^p_m = sgn(Σ_t w_t * τ^p_t)
    
    Args:
        weighted_task_vectors: Stacked weighted task vectors, shape (num_models, ...)
    
    Returns:
        Elected sign vector with values +1, 0, or -1
    """
    # Sum weighted task vectors along the first dimension
    sum_vector = weighted_task_vectors.sum(dim=0)
    
    # Return sign of the sum
    return torch.sign(sum_vector)


def disjoint_merge(
    task_vectors: Tensor,
    elected_signs: Tensor,
    weights: Tensor
) -> Tensor:
    """
    Merge task vectors by keeping only values whose signs match the elected sign.
    
    According to TIES paper: τ^p_m = mean(τ^p_t | sign(τ^p_t) == γ^p_m)
    
    Args:
        task_vectors: Stacked trimmed task vectors, shape (num_models, ...)
        elected_signs: Elected sign vector, shape (...)
        weights: Weights for each model, shape (num_models,)
    
    Returns:
        Merged task vector
    """
    # Get signs of each task vector
    signs = torch.sign(task_vectors)
    
    # Create mask: True where sign matches elected sign
    # Expand elected_signs to match task_vectors shape
    elected_signs_expanded = elected_signs.unsqueeze(0)
    mask = (signs == elected_signs_expanded)
    
    # Apply mask to task vectors
    masked_vectors = task_vectors * mask
    
    # Compute weighted sum of masked vectors
    # Expand weights to match task_vectors shape
    weights_expanded = weights.view(-1, *([1] * (task_vectors.dim() - 1)))
    weighted_sum = (masked_vectors * weights_expanded).sum(dim=0)
    
    # Compute sum of weights for normalization
    weight_sum = (weights_expanded * mask).sum(dim=0)
    
    # Avoid division by zero
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
    
    # Return weighted average
    return weighted_sum / weight_sum


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
    
    Parameters:
    - base_model: Base model tensor (pre-trained model)
    - *tensors: Variable number of fine-tuned model tensors to merge
    - json_params: JSON string containing configuration
      Example: {
        "lambda": 1.0,
        "global": {
          "weights": "1.0, 1.0, 1.0",
          "densities": "0.2, 0.2, 0.2"
        },
        "IN": {
          "weights": "0.5, 0.3, 0.2",
          "densities": "0.3, 0.2, 0.1"
        }
      }
    
    JSON Parameters:
    - lambda (global): Scaling parameter (default: 1.0)
    - weights (per-block): Comma-separated weights for each model (default: "1.0" for all)
    - densities (per-block): Comma-separated densities for each model, 0.0-1.0 (default: "0.2" for all)
    """
    global _weight_logger
    
    if len(tensors) == 0:
        return base_model
    
    params: Dict[str, Any] = parse_json_params(json_params)
    
    # Get global lambda parameter
    lambda_val: float = params.get("lambda", 1.0)
    
    # Get default global weights and densities
    global_config = params.get("global", {})
    if isinstance(global_config, str):
        # Legacy format: just weights string
        global_weights_str = global_config
        global_densities_str = "0.2"
    else:
        global_weights_str = global_config.get("weights", "1.0")
        global_densities_str = global_config.get("densities", "0.2")
    
    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    
    # Get block-specific config or use global
    block_config = params.get(user_block_name, {})
    if isinstance(block_config, str):
        # Legacy format: just weights string
        weights_str = block_config
        densities_str = global_densities_str
    else:
        weights_str = block_config.get("weights", global_weights_str)
        densities_str = block_config.get("densities", global_densities_str)
    
    # Log weights if they have changed
    config_str = f"weights={weights_str}, densities={densities_str}, lambda={lambda_val}"
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
    # First, apply weights to task vectors
    weights_expanded = weights_tensor.view(-1, *([1] * (task_vectors_stacked.dim() - 1)))
    weighted_task_vectors = task_vectors_stacked * weights_expanded
    
    # Elect signs: γ^p_m = sgn(Σ_t w_t * τ^p_t)
    elected_signs = elect_sign(weighted_task_vectors)
    
    # Step 3: Disjoint merge
    # Merge only values whose signs match the elected sign
    merged_task_vector = disjoint_merge(task_vectors_stacked, elected_signs, weights_tensor)
    
    # Step 4: Compute final result: θ_m = θ_init + λ * τ_m
    result = base_model + lambda_val * merged_task_vector
    
    # Final validation
    return validate_result(result, key)


import torch
import logging
from typing import Dict, Any
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .lib.parse_utils import parse_alphas, parse_json_params
from .lib.block_utils import get_user_block_name
from .lib.tensor_utils import fix_non_finite_tensors, validate_result
from .lib.weight_logger import WeightLogger
from .ties_json import elect_sign, disjoint_merge

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global weight logger instance
_weight_logger = WeightLogger()


def magprune_task_vector(task_vector: Tensor, density: float, gamma: float) -> Tensor:
    """
    Prune task vector using MAGPRUNE (magnitude-based pruning) from DELLA.
    
    MAGPRUNE removes:
    - (1-density-gamma) fraction of largest magnitude parameters
    - gamma fraction of smallest magnitude parameters
    - Keeps the remaining density fraction of parameters
    
    Edge case: If gamma >= 1.0 - density, only remove (1-density) largest magnitude parameters.
    
    Args:
        task_vector: Task vector to prune
        density: Fraction of parameters to keep (0.0-1.0)
        gamma: Fraction of smallest magnitude parameters to remove (0.0-1.0)
    
    Returns:
        Pruned task vector
    """
    if density <= 0.0:
        return torch.zeros_like(task_vector)
    if density >= 1.0:
        return task_vector
    
    num_elements = task_vector.numel()
    
    # Calculate how many parameters to keep
    num_to_keep = int(density * num_elements)
    
    if num_to_keep <= 0:
        return torch.zeros_like(task_vector)
    if num_to_keep >= num_elements:
        return task_vector
    
    # Compute absolute values for magnitude-based pruning
    abs_values = torch.abs(task_vector)
    abs_values_flat = abs_values.flatten()
    
    # Edge case: if gamma >= 1.0 - density, only remove (1-density) largest
    if gamma >= 1.0 - density:
        # Remove only (1-density) largest magnitude parameters
        num_to_remove = num_elements - num_to_keep
        if num_to_remove > 0:
            k = num_to_remove + 1
            threshold = torch.kthvalue(abs_values_flat, k).values
            mask = abs_values < threshold
        else:
            mask = torch.ones_like(abs_values, dtype=torch.bool)
    else:
        # Normal case: remove largest and smallest
        num_largest_to_remove = int((1.0 - density - gamma) * num_elements)
        num_smallest_to_remove = int(gamma * num_elements)
        
        # Remove largest magnitude parameters
        if num_largest_to_remove > 0:
            k_largest = num_elements - num_largest_to_remove + 1
            threshold_largest = torch.kthvalue(abs_values_flat, k_largest).values
            mask_largest = abs_values < threshold_largest
        else:
            mask_largest = torch.ones_like(abs_values, dtype=torch.bool)
        
        # Remove smallest magnitude parameters
        if num_smallest_to_remove > 0:
            k_smallest = num_smallest_to_remove + 1
            threshold_smallest = torch.kthvalue(abs_values_flat, k_smallest).values
            mask_smallest = abs_values > threshold_smallest
        else:
            mask_smallest = torch.ones_like(abs_values, dtype=torch.bool)
        
        # Combine masks: keep parameters that are not in largest or smallest
        # This ensures we keep exactly density fraction of parameters
        mask = mask_largest & mask_smallest
    
    return task_vector * mask


def della_merge_linear(
    task_vectors: Tensor,
    weights: Tensor,
) -> Tensor:
    """
    Merge pruned task vectors using weighted average (linear merge).
    
    Args:
        task_vectors: Stacked pruned task vectors, shape (num_models, ...)
        weights: Weights for each model, shape (num_models,)
    
    Returns:
        Merged task vector
    """
    # Expand weights to match task_vectors shape
    weights_expanded = weights.view(-1, *([1] * (task_vectors.dim() - 1)))
    
    # Weighted sum
    weighted_sum = (task_vectors * weights_expanded).sum(dim=0)
    weight_sum = weights.sum()
    
    # Avoid division by zero
    if weight_sum == 0:
        return torch.zeros_like(task_vectors[0])
    
    return weighted_sum / weight_sum


@merge_method(identifier="🔨_della_linear_with_json", register=True)
def della_linear_with_json(
        base_model: Parameter(Tensor),
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using the DELLA algorithm with linear merging,
    with the ability to set all parameters via a JSON string.
    
    DELLA uses MAGPRUNE (magnitude-based pruning) to address interference:
    1. Remove (1-density-gamma) fraction of largest magnitude parameters
    2. Remove gamma fraction of smallest magnitude parameters
    3. Keep the remaining density fraction of parameters
    4. Linear weighted averaging of pruned task vectors
    
    Parameters:
    - base_model: Base model tensor (pre-trained model)
    - *tensors: Variable number of fine-tuned model tensors to merge
    - json_params: JSON string containing configuration
      Example: {
        "lambda": 1.0,
        "global": {
          "weights": "1.0, 1.0, 1.0",
          "densities": "0.2, 0.2, 0.2",
          "gammas": "0.01, 0.01, 0.01"
        },
        "IN": {
          "weights": "0.5, 0.3, 0.2",
          "densities": "0.3, 0.2, 0.1",
          "gammas": "0.02, 0.01, 0.01"
        }
      }
    
    JSON Parameters:
    - lambda: Scaling parameter (default: 1.0)
    - weights: Comma-separated weights for each model (default: "1.0" for all)
    - densities: Comma-separated densities for each model, 0.0-1.0 (default: "0.2" for all)
    - gammas: Comma-separated gammas for each model, 0.0-1.0 (default: "0.01" for all)
    """
    global _weight_logger
    
    if len(tensors) == 0:
        return base_model
    
    params: Dict[str, Any] = parse_json_params(json_params)
    
    # Get global lambda parameter
    lambda_val: float = params.get("lambda", 1.0)
    
    # Get default global weights, densities, and gammas
    global_config = params.get("global", {})
    if isinstance(global_config, str):
        # Legacy format: just weights string
        global_weights_str = global_config
        global_densities_str = "0.2"
        global_gammas_str = "0.01"
    else:
        global_weights_str = global_config.get("weights", "1.0")
        global_densities_str = global_config.get("densities", "0.2")
        global_gammas_str = global_config.get("gammas", "0.01")
    
    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    
    # Get block-specific config or use global
    block_config = params.get(user_block_name, {})
    if isinstance(block_config, str):
        # Legacy format: just weights string
        weights_str = block_config
        densities_str = global_densities_str
        gammas_str = global_gammas_str
    else:
        weights_str = block_config.get("weights", global_weights_str)
        densities_str = block_config.get("densities", global_densities_str)
        gammas_str = block_config.get("gammas", global_gammas_str)
    
    # Log weights if they have changed
    config_str = f"weights={weights_str}, densities={densities_str}, gammas={gammas_str}, lambda={lambda_val}"
    _weight_logger.log_weights(user_block_name, config_str, "della_linear_with_json")
    
    # Parse weights, densities, and gammas
    weight_values = parse_alphas(weights_str)
    density_values = parse_alphas(densities_str)
    gamma_values = parse_alphas(gammas_str)
    
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
    if len(gamma_values) != num_models:
        raise ValueError(
            f"The number of gammas for block '{user_block_name}' ({len(gamma_values)}) "
            f"does not match the number of models ({num_models})."
        )
    
    # Validate density and gamma values
    for i, d in enumerate(density_values):
        if d < 0.0 or d > 1.0:
            raise ValueError(
                f"Density value {d} for model {i} in block '{user_block_name}' "
                f"must be between 0.0 and 1.0."
            )
    
    for i, g in enumerate(gamma_values):
        if g < 0.0 or g > 1.0:
            raise ValueError(
                f"Gamma value {g} for model {i} in block '{user_block_name}' "
                f"must be between 0.0 and 1.0."
            )
        # Check that density + gamma <= 1.0 (with some tolerance for floating point)
        if density_values[i] + g > 1.0 + 1e-6:
            raise ValueError(
                f"For model {i} in block '{user_block_name}': "
                f"density ({density_values[i]}) + gamma ({g}) must be <= 1.0."
            )
    
    # Convert to tensors
    weights_tensor = torch.tensor(weight_values, device=base_model.device, dtype=base_model.dtype)
    density_tensor = torch.tensor(density_values, device=base_model.device, dtype=base_model.dtype)
    gamma_tensor = torch.tensor(gamma_values, device=base_model.device, dtype=base_model.dtype)
    
    # Fix non-finite tensors
    base_model = fix_non_finite_tensors((base_model,), key)[0]
    tensors = fix_non_finite_tensors(tensors, key)
    
    # Step 1: Compute task vectors and prune them with MAGPRUNE
    task_vectors = []
    for i, tensor in enumerate(tensors):
        # Compute task vector: τ_t = θ_t - θ_init
        task_vector = tensor - base_model
        
        # MAGPRUNE task vector
        density = density_tensor[i].item()
        gamma = gamma_tensor[i].item()
        pruned_task_vector = magprune_task_vector(task_vector, density, gamma)
        
        task_vectors.append(pruned_task_vector)
    
    if not task_vectors:
        return base_model
    
    # Stack task vectors: shape (num_models, ...)
    task_vectors_stacked = torch.stack(task_vectors, dim=0)
    
    # Step 2: Linear merge (weighted average)
    merged_task_vector = della_merge_linear(task_vectors_stacked, weights_tensor)
    
    # Step 3: Compute final result: θ_m = θ_init + λ * τ_m
    result = base_model + lambda_val * merged_task_vector
    
    # Final validation
    return validate_result(result, key)


@merge_method(identifier="🔨_della_with_json", register=True)
def della_with_json(
        base_model: Parameter(Tensor),
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using the DELLA algorithm with TIES merging,
    with the ability to set all parameters via a JSON string.
    
    DELLA-TIES combines MAGPRUNE with TIES sign consensus:
    1. Remove (1-density-gamma) fraction of largest magnitude parameters
    2. Remove gamma fraction of smallest magnitude parameters
    3. Keep the remaining density fraction of parameters
    4. Elect signs to resolve sign conflicts (TIES)
    5. Disjoint merging (only merging values with matching signs)
    
    Parameters:
    - base_model: Base model tensor (pre-trained model)
    - *tensors: Variable number of fine-tuned model tensors to merge
    - json_params: JSON string containing configuration
      Example: {
        "lambda": 1.0,
        "vote_sgn": false,
        "global": {
          "weights": "1.0, 1.0, 1.0",
          "densities": "0.2, 0.2, 0.2",
          "gammas": "0.01, 0.01, 0.01",
          "vote_sgn": false
        },
        "IN": {
          "weights": "0.5, 0.3, 0.2",
          "densities": "0.3, 0.2, 0.1",
          "gammas": "0.02, 0.01, 0.01"
        }
      }
    
    JSON Parameters:
    - lambda: Scaling parameter (default: 1.0)
    - weights: Comma-separated weights for each model (default: "1.0" for all)
    - densities: Comma-separated densities for each model, 0.0-1.0 (default: "0.2" for all)
    - gammas: Comma-separated gammas for each model, 0.0-1.0 (default: "0.01" for all)
    - vote_sgn: If True, use sum of vectors for sign election; if False, use sum of signs (default: False)
    """
    global _weight_logger
    
    if len(tensors) == 0:
        return base_model
    
    params: Dict[str, Any] = parse_json_params(json_params)
    
    # Get global lambda parameter
    lambda_val: float = params.get("lambda", 1.0)
    
    # Get default global weights, densities, and gammas
    global_config = params.get("global", {})
    if isinstance(global_config, str):
        # Legacy format: just weights string
        global_weights_str = global_config
        global_densities_str = "0.2"
        global_gammas_str = "0.01"
        global_vote_sgn = False
    else:
        global_weights_str = global_config.get("weights", "1.0")
        global_densities_str = global_config.get("densities", "0.2")
        global_gammas_str = global_config.get("gammas", "0.01")
        global_vote_sgn = global_config.get("vote_sgn", params.get("vote_sgn", False))
    
    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    
    # Get block-specific config or use global
    block_config = params.get(user_block_name, {})
    if isinstance(block_config, str):
        # Legacy format: just weights string
        weights_str = block_config
        densities_str = global_densities_str
        gammas_str = global_gammas_str
        vote_sgn = global_vote_sgn
    else:
        weights_str = block_config.get("weights", global_weights_str)
        densities_str = block_config.get("densities", global_densities_str)
        gammas_str = block_config.get("gammas", global_gammas_str)
        vote_sgn = block_config.get("vote_sgn", global_vote_sgn)
    
    # Log weights if they have changed
    features_str = ""
    if vote_sgn:
        features_str = ", features=[vote_sgn]"
    config_str = f"weights={weights_str}, densities={densities_str}, gammas={gammas_str}, lambda={lambda_val}{features_str}"
    _weight_logger.log_weights(user_block_name, config_str, "della_with_json")
    
    # Parse weights, densities, and gammas
    weight_values = parse_alphas(weights_str)
    density_values = parse_alphas(densities_str)
    gamma_values = parse_alphas(gammas_str)
    
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
    if len(gamma_values) != num_models:
        raise ValueError(
            f"The number of gammas for block '{user_block_name}' ({len(gamma_values)}) "
            f"does not match the number of models ({num_models})."
        )
    
    # Validate density and gamma values
    for i, d in enumerate(density_values):
        if d < 0.0 or d > 1.0:
            raise ValueError(
                f"Density value {d} for model {i} in block '{user_block_name}' "
                f"must be between 0.0 and 1.0."
            )
    
    for i, g in enumerate(gamma_values):
        if g < 0.0 or g > 1.0:
            raise ValueError(
                f"Gamma value {g} for model {i} in block '{user_block_name}' "
                f"must be between 0.0 and 1.0."
            )
        # Check that density + gamma <= 1.0 (with some tolerance for floating point)
        if density_values[i] + g > 1.0 + 1e-6:
            raise ValueError(
                f"For model {i} in block '{user_block_name}': "
                f"density ({density_values[i]}) + gamma ({g}) must be <= 1.0."
            )
    
    # Validate boolean parameters
    if not isinstance(vote_sgn, bool):
        raise ValueError(f"vote_sgn must be a boolean, got {type(vote_sgn)}")
    
    # Convert to tensors
    weights_tensor = torch.tensor(weight_values, device=base_model.device, dtype=base_model.dtype)
    density_tensor = torch.tensor(density_values, device=base_model.device, dtype=base_model.dtype)
    gamma_tensor = torch.tensor(gamma_values, device=base_model.device, dtype=base_model.dtype)
    
    # Fix non-finite tensors
    base_model = fix_non_finite_tensors((base_model,), key)[0]
    tensors = fix_non_finite_tensors(tensors, key)
    
    # Step 1: Compute task vectors and prune them with MAGPRUNE
    task_vectors = []
    for i, tensor in enumerate(tensors):
        # Compute task vector: τ_t = θ_t - θ_init
        task_vector = tensor - base_model
        
        # MAGPRUNE task vector
        density = density_tensor[i].item()
        gamma = gamma_tensor[i].item()
        pruned_task_vector = magprune_task_vector(task_vector, density, gamma)
        
        task_vectors.append(pruned_task_vector)
    
    if not task_vectors:
        return base_model
    
    # Stack task vectors: shape (num_models, ...)
    task_vectors_stacked = torch.stack(task_vectors, dim=0)
    
    # Step 2: Elect signs (TIES)
    elected_signs = elect_sign(task_vectors_stacked, weights_tensor, vote_sgn=vote_sgn)
    
    # Step 3: Disjoint merge (TIES)
    merged_task_vector = disjoint_merge(
        task_vectors_stacked,
        elected_signs,
        weights_tensor,
        apply_stock=False,
        apply_median=False,
        cos_eps=1e-6,
        eps=1e-6,
        maxiter=100,
        ftol=1e-20
    )
    
    # Step 4: Compute final result: θ_m = θ_init + λ * τ_m
    result = base_model + lambda_val * merged_task_vector
    
    # Final validation
    return validate_result(result, key)


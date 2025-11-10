import torch
import logging
from typing import Dict, Any
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .lib.parse_utils import parse_alphas, parse_json_params
from .lib.block_utils import get_user_block_name
from .lib.tensor_utils import fix_non_finite_tensors, validate_result
from .lib.weight_logger import WeightLogger

# Set up logging to provide more informative messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global weight logger instance
_weight_logger = WeightLogger()


@merge_method(identifier="🔨_tensor_sum_with_json", register=True)
def tensor_sum_with_json(
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using a tensor sum algorithm with non-normalized weights,
    with the ability to set all parameters via a JSON string.
    
    The tensor_sum method computes: result = alpha1 * tensor1 + alpha2 * tensor2 + ... + alphaN * tensorN
    where weights are NOT normalized (sum of alphas does NOT need to equal 1).
    
    This allows for more flexible merging where you can amplify or reduce specific models,
    perform add-difference operations (A + α·(B - C)), or layer-specific merging.
    Unlike weight_sum, this method does NOT normalize weights, giving you full control
    over the scale of the result.
    
    Parameters:
    - *tensors: Variable number of tensors to merge
    - json_params: JSON string containing weights for different blocks
      Example: {"global": "0.5, 0.3, 0.2", "IN": "1.0, 0.5, -0.2", "OUT": "0.8, 0.8, 0.4"}
      Weights are used directly WITHOUT normalization - their sum does NOT need to equal 1.
      This allows for add-difference operations and flexible scaling.
    """
    global _weight_logger

    params: Dict[str, Any] = parse_json_params(json_params)

    # Get default global weights (default to 1.0 for each tensor if not specified)
    global_alphas_str: str = params.get("global", "1.0")

    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    alphas_str = params.get(user_block_name, global_alphas_str)
    
    # Handle both string format (legacy) and dict format
    if isinstance(alphas_str, dict):
        alphas_str = alphas_str.get("weights", global_alphas_str)
    elif not isinstance(alphas_str, str):
        alphas_str = global_alphas_str

    # Log weights if they have changed
    _weight_logger.log_weights(user_block_name, alphas_str, "tensor_sum_with_json (NOT normalized)")

    # Parse the selected alphas
    alpha_values = parse_alphas(alphas_str)
    if len(alpha_values) != len(tensors):
        raise ValueError(f"The number of weights for block '{user_block_name}' does not match the number of models.")
    
    # Convert alphas to tensor on the same device as the first tensor
    # NOTE: We do NOT normalize these weights - this is the key difference from weight_sum
    alphas_tensor = torch.tensor(alpha_values, device=tensors[0].device, dtype=tensors[0].dtype)

    # Fix non-finite tensors
    tensors = fix_non_finite_tensors(tensors, key)

    # If only one tensor, return it directly (multiplied by its weight)
    if len(tensors) == 1:
        return alphas_tensor[0] * tensors[0]

    # Compute tensor sum WITHOUT normalization: result = sum(alpha_i * tensor_i)
    # where sum(alpha_i) does NOT need to equal 1
    result = torch.zeros_like(tensors[0])
    for alpha, tensor in zip(alphas_tensor, tensors):
        result += alpha * tensor

    # Final check before returning the result
    return validate_result(result, key)


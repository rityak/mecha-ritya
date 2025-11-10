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


@merge_method(identifier="🔨_weight_sum_with_json", register=True)
def weight_sum_with_json(
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using a normalized weighted sum algorithm,
    with the ability to set all parameters via a JSON string.
    
    The weight_sum method computes: result = sum(normalized_alpha_i * tensor_i)
    where normalized_alpha_i are weights normalized so their sum equals 1.
    This ensures the result behaves like an average/interpolation between models.
    
    This is a straightforward linear combination with normalization, unlike more complex methods
    like Karcher mean that use iterative algorithms on manifolds.
    
    Parameters:
    - *tensors: Variable number of tensors to merge
    - json_params: JSON string containing weights for different blocks
      Example: {"global": "0.5, 0.3, 0.2", "IN": "0.6, 0.2, 0.2", "OUT": "0.4, 0.4, 0.2"}
      Weights are automatically normalized (sum = 1) for predictable averaging behavior.
    """
    global _weight_logger

    params: Dict[str, Any] = parse_json_params(json_params)

    # Get default global weights
    global_alphas_str: str = params.get("global", "1.0")

    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    alphas_str = params.get(user_block_name, global_alphas_str)

    # Log weights if they have changed
    _weight_logger.log_weights(user_block_name, alphas_str, "weight_sum_with_json")

    # Parse the selected alphas
    alpha_values = parse_alphas(alphas_str)
    if len(alpha_values) != len(tensors):
        raise ValueError(f"The number of weights for block '{user_block_name}' does not match the number of models.")
    
    # Convert alphas to tensor on the same device as the first tensor
    alphas_tensor = torch.tensor(alpha_values, device=tensors[0].device, dtype=tensors[0].dtype)
    
    # Normalize weights so their sum equals 1 (this is the key characteristic of weight sum)
    alpha_sum = alphas_tensor.sum()
    if abs(alpha_sum.item()) < 1e-10:
        logging.warning(f"Sum of weights for block '{user_block_name}' is near zero. Using equal weights.")
        normalized_alphas = torch.ones_like(alphas_tensor) / len(tensors)
    else:
        normalized_alphas = alphas_tensor / alpha_sum

    # Fix non-finite tensors
    tensors = fix_non_finite_tensors(tensors, key)

    # If only one tensor, return it directly
    if len(tensors) == 1:
        return tensors[0]

    # Compute normalized weighted sum: result = sum(normalized_alpha_i * tensor_i)
    # where sum(normalized_alpha_i) = 1
    result = torch.zeros_like(tensors[0])
    for alpha, tensor in zip(normalized_alphas, tensors):
        result += alpha * tensor

    # Final check before returning the result
    return validate_result(result, key)


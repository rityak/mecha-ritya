"""
Utilities for handling tensor operations and validation.
"""
import torch
from typing import Tuple
from torch import Tensor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def fix_non_finite_tensors(tensors: Tuple[Tensor, ...], key: str = "n/a") -> Tuple[Tensor, ...]:
    """
    Handles input data: checks for non-finite values and tries to replace them
    with a value from another tensor before falling back to zeros.
    
    Args:
        tensors: Tuple of tensors to check and fix
        key: Optional key for logging purposes
    
    Returns:
        Tuple of corrected tensors
    """
    corrected_tensors = list(tensors)
    for i, t in enumerate(tensors):
        if torch.isnan(t).any() or torch.isinf(t).any():
            logging.warning(
                f"Non-finite values detected in tensor {i} for key '{key}'. Attempting to find a valid replacement.")
            replacement_found = False
            for j, other_t in enumerate(tensors):
                if i != j and not (torch.isnan(other_t).any() or torch.isinf(other_t).any()):
                    corrected_tensors[i] = other_t.clone()
                    logging.info(f"Successfully replaced tensor {i} with a value from tensor {j}.")
                    replacement_found = True
                    break
            if not replacement_found:
                corrected_tensors[i] = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                logging.warning(
                    f"No valid replacement found. All tensors for this key may be non-finite. Replacing with zeros.")
    return tuple(corrected_tensors)


def validate_result(result: Tensor, key: str = "n/a") -> Tensor:
    """
    Final check before returning the result. Fixes non-finite values instead of returning zero tensor.
    
    Args:
        result: Tensor to validate
        key: Optional key for logging purposes
    
    Returns:
        Validated tensor with non-finite values replaced
    """
    if torch.isnan(result).any() or torch.isinf(result).any():
        nan_count = torch.isnan(result).sum().item()
        inf_count = torch.isinf(result).sum().item()
        logging.warning(f"The final result contains non-finite values for key '{key}': {nan_count} NaN, {inf_count} Inf. Fixing with nan_to_num.")
        # Fix non-finite values instead of returning zero tensor
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


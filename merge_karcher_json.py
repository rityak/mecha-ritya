import json
import torch
import math
from typing import Tuple, Dict, Any, List
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return
import logging

# Set up logging to provide more informative messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Global dictionary to store the last logged weights for each block to avoid redundant logging.
_last_logged_weights = {}


def get_user_block_name(key: str, params: Dict[str, Any]) -> str:
    """
    Dynamically determines the user-friendly block name based on the model key.
    This function prioritizes specific block names from the JSON, then group names, then falls back to 'global'.
    """
    # 1. Check for specific block number names (e.g., IN00)
    parts = key.split('.')
    if "input_blocks" in parts:
        try:
            index = parts[parts.index("input_blocks") + 1]
            block_name = f"IN{int(index):02}"
            if block_name in params:
                return block_name
        except (ValueError, IndexError):
            pass  # Fallback to group name
    elif "middle_block" in parts:
        if "M00" in params:
            return "M00"
    elif "output_blocks" in parts:
        try:
            index = parts[parts.index("output_blocks") + 1]
            block_name = f"OUT{int(index):02}"
            if block_name in params:
                return block_name
        except (ValueError, IndexError):
            pass  # Fallback to group name

    # 2. Check for broader group names (e.g., IN, OUT)
    if "input_blocks" in parts:
        if "IN" in params:
            return "IN"
    elif "middle_block" in parts:
        if "M" in params:
            return "M"
    elif "output_blocks" in parts:
        if "OUT" in params:
            return "OUT"
    elif key.startswith("conditioner.embedders.0"):
        if "CLIP_L" in params:
            return "CLIP_L"
    elif key.startswith("conditioner.embedders.1"):
        if "CLIP_G" in params:
            return "CLIP_G"
    elif key.startswith("model.diffusion_model.time_embed"):
        if "time_embed" in params:
            return "time_embed"
    elif key.startswith("model.diffusion_model.label_emb"):
        if "label_emb" in params:
            return "label_emb"

    # 3. Fallback to global
    return "global"


def parse_alphas(alphas_str: str) -> List[float]:
    """Parses a string of alphas into a list of numbers."""
    try:
        return [float(a.strip()) for a in alphas_str.split(',')]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid weight format: '{alphas_str}'. Weights must be numbers separated by commas.")


@merge_method(identifier="karcher_mean_with_json", register=True)
def karcher_mean_with_json(
        *tensors: Parameter(Tensor),
        json_params: Parameter(str),
        **kwargs
) -> Return(Tensor):
    """
    Merges several model tensors using the Karcher mean (Riemannian mean) algorithm,
    with the ability to set all parameters via a JSON string.
    """
    global _last_logged_weights

    try:
        params: Dict[str, Any] = json.loads(json_params) if json_params.strip() else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Set default values
    max_iter: int = params.get("max_iter", 10)
    tol: float = params.get("tol", 1e-5)
    global_alphas_str: str = params.get("global", "1.0")

    # Get the current tensor key and determine the block name
    key = kwargs.get("key", "n/a")
    user_block_name = get_user_block_name(key, params)
    alphas_str = params.get(user_block_name, global_alphas_str)

    # Check if the weights for the current block have changed, and log if they have.
    if _last_logged_weights.get(user_block_name) != alphas_str:
        logging.info(f"Using weights '{alphas_str}' for block '{user_block_name}'.")
        _last_logged_weights[user_block_name] = alphas_str

    # Parse the selected alphas
    alpha_values = parse_alphas(alphas_str)
    if len(alpha_values) != len(tensors):
        raise ValueError(f"The number of weights for block '{user_block_name}' does not match the number of models.")
    alphas_tensor = torch.tensor(alpha_values, device=tensors[0].device)

    # Handle input data: check for non-finite values and try to replace them
    # with a value from another tensor before falling back to zeros.
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
    tensors = tuple(corrected_tensors)

    if len(tensors) == 1:
        return tensors[0]

    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val < tol:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))

    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    valid_alphas = alphas_tensor[valid_indices]
    alpha_sum = valid_alphas.sum()
    normalized_alphas = valid_alphas / alpha_sum
    valid_units = [units[i] for i in valid_indices]

    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    for i in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Clamp the dot product to a safe range to prevent arccos from returning NaN
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -0.999999, 0.999999)
            theta = torch.arccos(dot)
            if theta.item() < tol:
                continue
            sin_theta = torch.sin(theta)
            # Protect against division by very small numbers
            if abs(sin_theta.item()) < tol:
                continue
            T += a * (theta / sin_theta) * (ui - dot * u)

        # Check T for NaN/Inf after computation. If found, break iterations.
        if torch.isnan(T).any() or torch.isinf(T).any():
            logging.warning(f"Non-finite values in intermediate result (T) for key '{key}'. Breaking iterations.")
            T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
            break

        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break

        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        # Protect against division by very small numbers
        if abs(sin_norm_T.item()) < tol:
            break

        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)
        u_norm = torch.linalg.norm(u.float())

        # Check u for NaN/Inf after updating
        if torch.isnan(u).any() or torch.isinf(u).any():
            logging.warning(f"Non-finite values in the updated vector (u) for key '{key}'. Breaking iterations.")
            u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            break

        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)

    s = sum(a.item() * n for a, n in zip(alphas_tensor, norms))

    # Final check before returning the result
    result = s * u
    if torch.isnan(result).any() or torch.isinf(result).any():
        logging.warning(f"The final result contains non-finite values for key '{key}'. Returning a zero tensor.")
        return torch.zeros_like(tensors[0])
    return result
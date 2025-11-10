import torch
import math
import logging
from typing import Tuple, Dict, Any
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .lib.parse_utils import parse_alphas
from .lib.tensor_utils import fix_non_finite_tensors, validate_result

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_block_name(key):
    """Determines the block name for a key based on patterns."""
    if key.startswith("model.diffusion_model.input_blocks"):
        return "input_blocks"
    elif key.startswith("model.diffusion_model.middle_block"):
        return "middle_block"
    elif key.startswith("model.diffusion_model.output_blocks"):
        return "output_blocks"
    elif key.startswith("model.diffusion_model.out"):
        return "output_blocks"
    elif key.startswith("conditioner.embedders.0"):
        return "clip_l"
    elif key.startswith("conditioner.embedders.1"):
        return "clip_g"
    elif key.startswith("model.diffusion_model.time_embed"):
        return "time_embed"
    elif key.startswith("model.diffusion_model.label_emb"):
        return "label_emb"
    return None


@merge_method(identifier="🔨_karcher_mean_with_blocks", register=True)
def karcher_mean_with_blocks(
        *tensors: Parameter(Tensor),
        alpha_global: Parameter(str),
        alpha_clip_l: Parameter(str) = None,
        alpha_clip_g: Parameter(str) = None,
        alpha_in: Parameter(str) = None,
        alpha_mid: Parameter(str) = None,
        alpha_out: Parameter(str) = None,
        alpha_embed_time: Parameter(str) = None,
        alpha_embed_label: Parameter(str) = None,
        max_iter: Parameter(int) = 10,
        tol: Parameter(float) = 1e-5,
        **kwargs) -> Return(Tensor):
    """
    Merges several model tensors using the Karcher mean (Riemannian mean) algorithm,
    with the ability to set alphas for different blocks.

    Parameters:
    - *tensors: A variable number of tensors to merge.
    - alpha_global: A string with global weights, separated by commas (e.g., "0.5, 0.25, 0.25").
    - alpha_clip_l: A string with weights for the clip_l block (optional).
    - alpha_clip_g: A string with weights for the clip_g block (optional).
    - alpha_in: A string with weights for the input_blocks block (optional).
    - alpha_mid: A string with weights for the middle_block block (optional).
    - alpha_out: A string with weights for the output_blocks block (optional).
    - alpha_embed_time: A string with weights for the time_embed block (optional).
    - alpha_embed_label: A string with weights for the label_emb block (optional).
    - max_iter: The maximum number of iterations for the algorithm.
    - tol: The tolerance for convergence.
    """
    # Parse global alphas
    alpha_global_values = parse_alphas(alpha_global)
    if len(alpha_global_values) != len(tensors):
        raise ValueError("The number of global weights must match the number of tensors.")

    key = kwargs.get("key")
    block = get_block_name(key) if key else None

    if block == "clip_l" and alpha_clip_l:
        alphas_str = alpha_clip_l
    elif block == "clip_g" and alpha_clip_g:
        alphas_str = alpha_clip_g
    elif block == "input_blocks" and alpha_in:
        alphas_str = alpha_in
    elif block == "middle_block" and alpha_mid:
        alphas_str = alpha_mid
    elif block == "output_blocks" and alpha_out:
        alphas_str = alpha_out
    elif block == "time_embed" and alpha_embed_time:
        alphas_str = alpha_embed_time
    elif block == "label_emb" and alpha_embed_label:
        alphas_str = alpha_embed_label
    else:
        alphas_str = alpha_global

    # Parse the selected alphas
    alpha_values = parse_alphas(alphas_str)
    if len(alpha_values) != len(tensors):
        raise ValueError(f"The number of weights for block {block} must match the number of tensors.")
    alphas_tensor = torch.tensor(alpha_values, device=tensors[0].device)

    # Fix non-finite tensors
    tensors = fix_non_finite_tensors(tensors, key)

    if len(tensors) == 1:
        return tensors[0]

    # Calculate norms and unit vectors
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

    # Select valid indices where the norm exceeds the tolerance
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    # Prepare valid weights and unit vectors
    valid_alphas = alphas_tensor[valid_indices]
    alpha_sum = valid_alphas.sum()
    normalized_alphas = valid_alphas / alpha_sum
    valid_units = [units[i] for i in valid_indices]

    # Initialize u as a weighted arithmetic mean of the unit vectors
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    # Iteratively compute the Karcher mean
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

    # Apply global scale using all original norms
    s = sum(a.item() * n for a, n in zip(alphas_tensor, norms))

    # Final check before returning the result
    result = s * u
    return validate_result(result, key)
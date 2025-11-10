import torch
import math
import logging
from torch import Tensor
from sd_mecha import merge_method, Parameter, Return

from .lib.parse_utils import parse_alphas
from .lib.tensor_utils import fix_non_finite_tensors, validate_result

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


@merge_method(identifier="🔨_karcher_mean", register=True)
def karcher_mean(*tensors: Parameter(Tensor),
                 alphas: Parameter(str),
                 max_iter: Parameter(int) = 10,
                 tol: Parameter(float) = 1e-5,
                 **kwargs) -> Return(Tensor):
    """
    Merges several model tensors using the Karcher mean (Riemannian mean) algorithm.

    Parameters:
    - *tensors: A variable number of tensors to merge.
    - alphas: A string with weights, separated by commas (e.g., "0.5, 0.25, 0.25").
    - max_iter: The maximum number of iterations for the algorithm.
    - tol: The tolerance for convergence.
    """
    key = kwargs.get("key", "n/a")

    # Parse the alphas string into a list of numbers
    alpha_values = parse_alphas(alphas)

    if len(alpha_values) != len(tensors):
        raise ValueError("The number of weights must match the number of tensors.")
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
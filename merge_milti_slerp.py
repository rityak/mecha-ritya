from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import torch


def parse_weights(weights_str: str) -> list[float]:
    """Parses a comma-separated string of weights into a list of floats."""
    return [float(w.strip()) for w in weights_str.split(',')]


@merge_method(identifier="milti_slerp", register=True)
def multislerp(
        *tensors: Parameter(Tensor),
        weights: Parameter(str),
        base_model: Parameter(Tensor) = None,
        normalize_weights: Parameter(bool) = True,
        eps: Parameter(float) = 1e-8
) -> Return(Tensor):
    """Implements Multi-SLERP for merging multiple model tensors on a hypersphere.

    Parameters:
    - *tensors: Variable number of model tensors to merge.
    - weights: Comma-separated string of per-model weights (e.g., "0.5, 0.25, 0.25").
    - base_model: Optional base model tensor for task vector space operation.
    - normalize_weights: If True, normalize weights to sum to 1.
    - eps: Small constant for numerical stability.
    """
    # Parse and validate weights
    weight_values = parse_weights(weights)
    if len(weight_values) != len(tensors):
        raise ValueError("Number of weights must match number of input tensors")

    weights_tensor = torch.tensor(weight_values, dtype=torch.float32)
    if normalize_weights:
        weights_tensor = weights_tensor / (weights_tensor.sum() + eps)

    if len(tensors) == 1:
        return tensors[0]

    # Convert tensors to float for numerical stability
    tensors_float = [t.float() for t in tensors]

    # If base_model is provided, operate in task vector space (subtract base_model)
    if base_model is not None:
        tensors_float = [t - base_model.float() for t in tensors_float]

    # Compute weighted Euclidean mean as initial point
    mean = torch.zeros_like(tensors_float[0])
    for w, t in zip(weights_tensor, tensors_float):
        mean += w * t
    norm_mean = torch.linalg.norm(mean)
    if norm_mean.item() < eps:
        mean = tensors_float[0].clone()  # Fallback to first tensor if mean is zero
    else:
        mean = mean / norm_mean  # Normalize to unit vector

    # Project to tangent space at mean using logarithmic map
    tangent_vectors = []
    for t in tensors_float:
        t_norm = torch.linalg.norm(t)
        if t_norm.item() < eps:
            tangent_vectors.append(torch.zeros_like(t))
            continue
        t_unit = t / t_norm
        dot = torch.clamp(torch.dot(t_unit.flatten(), mean.flatten()), -1.0, 1.0)
        theta = torch.arccos(dot)
        if theta.item() < eps:
            tangent_vectors.append(torch.zeros_like(t))
        else:
            v = (theta / torch.sin(theta)) * (t_unit - dot * mean)
            tangent_vectors.append(v)

    # Perform weighted interpolation in tangent space
    interpolated_tangent = torch.zeros_like(tangent_vectors[0])
    for w, v in zip(weights_tensor, tangent_vectors):
        interpolated_tangent += w * v

    # Project back to hypersphere using exponential map
    norm_tangent = torch.linalg.norm(interpolated_tangent)
    if norm_tangent.item() < eps:
        result = mean
    else:
        cos_norm = torch.cos(norm_tangent)
        sin_norm = torch.sin(norm_tangent)
        result = cos_norm * mean + (sin_norm / norm_tangent) * interpolated_tangent

    # If base_model was used, add it back
    if base_model is not None:
        result = result + base_model.float()

    # Scale by average norm of input tensors
    avg_norm = sum(w.item() * torch.linalg.norm(t).item() for w, t in zip(weights_tensor, tensors_float))
    result = avg_norm * result

    return result.to(tensors[0].dtype)
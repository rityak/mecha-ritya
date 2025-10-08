from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import torch

def parse_weights(weights_str):
    """Parses a string with weights into a list of numbers."""
    return [float(w.strip()) for w in weights_str.split(',')]

@merge_method
def task_arithmetic(base_tensor: Parameter(Tensor), *tensors: Parameter(Tensor),
                    weights: Parameter(str), lambda_: Parameter(float) = 1.0) -> Return(Tensor):
    """Model merging using Task Arithmetic.

    Computes task vectors as the difference between each model and the base model,
    combines them with weights and adds to the base model with scaling.

    Parameters:
    - base_tensor: Base model tensor.
    - *tensors: Variable number of additional model tensors.
    - weights: String with weights separated by commas (e.g., "0.5, 0.25, 0.25").
    - lambda_: Global scaling coefficient for the summed task vector.
    """
    # Parse weights string
    weight_values = parse_weights(weights)
    if len(weight_values) != len(tensors):
        raise ValueError("Number of weights must match the number of additional tensors")

    # If no additional tensors, return base
    if not tensors:
        return base_tensor

    # Compute task vectors
    task_vectors = [tensor - base_tensor for tensor in tensors]

    # Weighted combination of task vectors
    weights_tensor = torch.tensor(weight_values, dtype=base_tensor.dtype, device=base_tensor.device)
    mixed_delta = torch.zeros_like(base_tensor)
    for tv, w in zip(task_vectors, weights_tensor):
        mixed_delta += w * tv

    # Scale the summed task vector
    mixed_delta *= lambda_

    # Add to base model
    return base_tensor + mixed_delta
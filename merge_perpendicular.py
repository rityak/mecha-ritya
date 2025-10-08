from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import torch


@merge_method(identifier="add_perpendicular_merge", register=True)
def add_perpendicular(
        a: Parameter(Tensor),
        b: Parameter(Tensor),
        c: Parameter(Tensor),
        alpha: Parameter(float) = 1.0,
) -> Return(Tensor):
    """
    Adds the perpendicular component of the difference vector between B and C
    to model A, using the original `perpendicular_component` formula.

    Parameters:
    - a: The base model tensor.
    - b: The model tensor with unique characteristics.
    - c: The common base model tensor for A and B.
    - alpha: The strength of the perpendicular component to add.
    """

    a_diff = a - c
    b_diff = b - c

    norm_a_diff = torch.linalg.norm(a_diff)

    # Calculate the perpendicular component using the original formula
    if norm_a_diff == 0:
        perp_component = b_diff
    else:
        res = b_diff - a_diff * (a_diff / norm_a_diff * (b_diff / norm_a_diff)).sum()
        perp_component = res

    return a + alpha * perp_component
import torch
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return

@merge_method(identifier="smooth_peak_merge", register=True, is_conversion=False)
def smooth_peak_merge(
    # These parameters are now individual torch.Tensor for the current key
    model_a: Parameter(torch.Tensor, merge_space="weight"),  # Weight tensor from model A for the current key
    model_b: Parameter(torch.Tensor, merge_space="weight"),  # Weight tensor from model B for the current key
    model_c: Parameter(torch.Tensor, merge_space="weight"),  # Weight tensor from model C for the current key
    
    # These parameters remain float, now with default values
    alpha: Parameter(float) = 0.5,  # Default value 0.5 for alpha
    smooth: Parameter(float) = 0.0, # Default value 0.0 for smooth
    
    # Add **kwargs to capture additional 'key' argument,
    # which sd-mecha passes for the current tensor.
    **kwargs,
) -> Return(torch.Tensor, merge_space="weight"): # Return value is the merged tensor
    """
    Performs smooth merging of individual tensors from models A, B, C for the current key.
    This method softens large differences between A/C and B/C,
    effectively smoothing 'peaks' in model parameter contributions.

    Arguments:
        model_a (Parameter(torch.Tensor, merge_space="weight")): Weight tensor from model A for the current key.
        model_b (Parameter(torch.Tensor, merge_space="weight")): Weight tensor from model B for the current key.
        model_c (Parameter(torch.Tensor, merge_space="weight")): Weight tensor from model C for the current key.
        alpha (Parameter(float)): Blending coefficient for A and B (from 0.0 to 1.0).
                       Higher alpha value favors model A. Default is 0.5.
        smooth (Parameter(float)): Non-negative coefficient for peak smoothing.
                        - If 0, standard weighted sum is performed (A * alpha + B * (1-alpha)).
                        - As smooth increases, large differences from C
                          are more strongly dampened, leading to 'smoother' merging. Default is 0.0.
        **kwargs: Includes 'key' (string name of current tensor), if needed.

    Returns:
        Return(torch.Tensor, merge_space="weight"): Merged tensor for the current key.
    """
    
    diff_ac = model_a - model_c
    diff_bc = model_b - model_c

    # Convert alpha and smooth to tensor type for correct operations with torch.where
    # If sd-mecha already passes them as tensors, this won't hurt.
    # If they are float, torch will convert them to tensors.
    alpha_tensor = torch.tensor(alpha, device=model_a.device)
    smooth_tensor = torch.tensor(smooth, device=model_a.device)

    smoothed_diff_ac = torch.where(
        torch.abs(diff_ac) > 1e-6,
        diff_ac / (1 + smooth_tensor * torch.abs(diff_ac)),
        diff_ac
    )
    smoothed_diff_bc = torch.where(
        torch.abs(diff_bc) > 1e-6,
        diff_bc / (1 + smooth_tensor * torch.abs(diff_bc)),
        diff_bc
    )

    blended_smoothed_diff = alpha_tensor * smoothed_diff_ac + (1 - alpha_tensor) * smoothed_diff_bc

    # Return the merged tensor directly for the current key
    return model_c + blended_smoothed_diff
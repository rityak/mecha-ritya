import torch
from torch import Tensor
from typing import Tuple, List
from sd_mecha.extensions.merge_methods import merge_method, Parameter, Return, StateDict
import math # For math.isclose

# Helper function to calculate individual k_threshold
def _calculate_individual_k_threshold(
    a: Tensor, 
    k: float, 
    min_sample_els: int, 
    sample_sz: int, 
    device: torch.device
) -> float:
    """
    Calculates the k-th percentile threshold for a tensor.
    Uses random sampling for large tensors (performance optimization).
    """
    # k specifies the fraction of *most significant* elements to keep.
    # torch.quantile takes a 'q' value (percentile from bottom).
    q_value = max(0.0, min(1.0, 1.0 - k))
    
    abs_a = a.abs().float()

    if abs_a.numel() == 0:
        return 0.0

    non_zero_abs_a = abs_a[abs_a > 1e-12] 
    if non_zero_abs_a.numel() == 0:
        return 0.0

    # Performance optimization: Sample for large tensors.
    if non_zero_abs_a.numel() > min_sample_els: 
        actual_sample_size = min(non_zero_abs_a.numel(), sample_sz) 
        indices = torch.randint(0, non_zero_abs_a.numel(), (actual_sample_size,), device=device)
        sampled_elements = non_zero_abs_a[indices]
        k_threshold = torch.quantile(sampled_elements, q_value)
    else:
        k_threshold = torch.quantile(non_zero_abs_a, q_value)
    
    return k_threshold.item() if isinstance(k_threshold, Tensor) else k_threshold

@merge_method(identifier="hierarchical_iterative_ties", register=True, is_conversion=False)
def hierarchical_iterative_ties(
    base: Parameter(torch.Tensor, merge_space="weight"),
    *models: Parameter(torch.Tensor, merge_space="weight"),
    top_k: Parameter(float) = 0.2,
    min_agr_cnt: Parameter(int) = 1,
    vot_sign: Parameter(bool) = False,
    min_sample_els: Parameter(int) = 1_000_000,
    sample_sz: Parameter(int) = 100_000, 
    max_itrs: Parameter(int) = 5,
    **kwargs,
) -> Return(Tensor, "delta"):
    """
    Performs an iterative, hierarchical consensus-based merge inspired by TIES.

    This method iteratively selects 'top k' deltas from *remaining* elements
    of each model, building a hierarchy of importance. Aggregation prioritizes
    elements by iteration level and consensus count.

    Args:
        base (Parameter(torch.Tensor, merge_space="weight")): The base model tensor.
        *models (Parameter(torch.Tensor, merge_space="weight")): Other model tensors.
        top_k (Parameter(float)): Fraction (0.0-1.0) of most significant deltas to select
        per iteration. Loop continues while cumulative sum <= 1.0.
        min_agr_cnt (Parameter(int)): Min number of models that must agree on an element's
        sign for it to be processed at a given hierarchical level.
        vot_sign (Parameter(bool)): If True, fallback average includes only models whose
        original delta sign matches consensus.
        min_sample_els (Parameter(int)): For performance: Min elements to trigger sampling for quantile.
        sample_sz (Parameter(int)): For performance: Number of elements to sample for quantile.
        max_itrs (Parameter(int)): Safety parameter: Max selection iterations.
        **kwargs: Includes 'key' (current tensor name).

    Returns:
        Return(Tensor, "delta"): The aggregated delta tensor.
    """
    if not models:
        return torch.zeros_like(base)

    device = base.device
    num_models = len(models)

    min_agr_cnt = max(0, min(min_agr_cnt, num_models))

    # 1. Initial Deltas Calculation
    original_deltas = [model.to(device) - base for model in models]
    cumulative_selected_mask_per_model = [torch.zeros_like(base, dtype=torch.bool, device=device) for _ in range(num_models)]
    hierarchical_selected_deltas_by_iteration = []

    # 2. Iterative Selection Loop (Building the Hierarchy)
    iteration_count = 0
    cumulative_top_k_sum_fraction = 0.0

    while cumulative_top_k_sum_fraction < 1.0 + 1e-6 and iteration_count < max_itrs:
        iteration_count += 1
        cumulative_top_k_sum_fraction += top_k

        current_iteration_selected_deltas_per_model = []
        has_any_model_remaining_deltas = False

        for model_idx in range(num_models):
            current_model_original_delta = original_deltas[model_idx]
            already_selected_mask_for_model = cumulative_selected_mask_per_model[model_idx]

            remaining_deltas_to_consider = torch.where(
                ~already_selected_mask_for_model, 
                current_model_original_delta, 
                torch.zeros_like(current_model_original_delta)
            )
            
            active_deltas_for_k_calc = remaining_deltas_to_consider[remaining_deltas_to_consider.abs() > 1e-12]

            selected_deltas_this_iter_for_model = torch.zeros_like(base, device=device)
            selected_mask_this_iter_for_model = torch.zeros_like(base, dtype=torch.bool, device=device)

            if active_deltas_for_k_calc.numel() > 0:
                has_any_model_remaining_deltas = True
                k_threshold_for_model = _calculate_individual_k_threshold(
                    active_deltas_for_k_calc, top_k, min_sample_els, sample_sz, device # <-- Использование сокращенных названий
                )

                selected_mask_this_iter_for_model = (current_model_original_delta.abs() >= k_threshold_for_model) & (~already_selected_mask_for_model)
                selected_deltas_this_iter_for_model = torch.where(
                    selected_mask_this_iter_for_model, 
                    current_model_original_delta, 
                    torch.zeros_like(current_model_original_delta)
                )
                cumulative_selected_mask_per_model[model_idx] |= selected_mask_this_iter_for_model
            
            current_iteration_selected_deltas_per_model.append(selected_deltas_this_iter_for_model)
        
        if has_any_model_remaining_deltas:
            hierarchical_selected_deltas_by_iteration.append(current_iteration_selected_deltas_per_model)
        else:
            break

    # 3. Global Consensus Sign Calculation
    all_accumulated_significant_deltas_sum = torch.zeros_like(base, device=device)
    for iter_deltas_per_model in hierarchical_selected_deltas_by_iteration:
        for model_selected_deltas in iter_deltas_per_model:
            all_accumulated_significant_deltas_sum += model_selected_deltas
    
    final_consensus_signs = torch.sign(all_accumulated_significant_deltas_sum)

    # 4. Hierarchical Aggregation
    final_delta_result = torch.zeros_like(base, device=device)
    elements_processed_in_aggregation_mask = torch.zeros_like(base, dtype=torch.bool, device=device)

    for iter_idx, iter_deltas_per_model in enumerate(hierarchical_selected_deltas_by_iteration):
        unprocessed_elements_mask = ~elements_processed_in_aggregation_mask

        current_iter_agreed_deltas_sum = torch.zeros_like(base, device=device)
        current_iter_agreed_counts = torch.zeros_like(base, dtype=torch.int, device=device)

        for model_idx in range(num_models):
            model_selected_deltas_this_iter = iter_deltas_per_model[model_idx]
            
            is_active_and_unprocessed = (model_selected_deltas_this_iter.abs() > 1e-12) & unprocessed_elements_mask
            sign_agrees_mask = (torch.sign(model_selected_deltas_this_iter) == final_consensus_signs)
            agreed_for_current_element_mask = is_active_and_unprocessed & sign_agrees_mask
            
            current_iter_agreed_deltas_sum += torch.where(agreed_for_current_element_mask, model_selected_deltas_this_iter, torch.zeros_like(base))
            current_iter_agreed_counts += torch.where(agreed_for_current_element_mask, torch.ones_like(base, dtype=torch.int), torch.zeros_like(base, dtype=torch.int))
        
        meet_agreement_threshold_mask = (current_iter_agreed_counts >= min_agr_cnt) & unprocessed_elements_mask 
        
        avg_agreed_delta_this_iter = torch.where(
            meet_agreement_threshold_mask & (current_iter_agreed_counts > 0),
            current_iter_agreed_deltas_sum / current_iter_agreed_counts.float(),
            torch.zeros_like(base)
        )
        
        final_delta_result = torch.where(meet_agreement_threshold_mask, avg_agreed_delta_this_iter, final_delta_result)
        elements_processed_in_aggregation_mask |= meet_agreement_threshold_mask

    # 5. Fallback (for still unprocessed elements)
    fallback_mask = ~elements_processed_in_aggregation_mask
    if fallback_mask.any():
        if vot_sign:
            sum_original_weights_for_fallback = base.clone() 
            count_for_fallback_avg = torch.ones_like(base, dtype=torch.float, device=device)
            
            for model_idx in range(num_models):
                current_model_original_delta = original_deltas[model_idx]
                
                sign_matches_consensus = (torch.sign(current_model_original_delta) == final_consensus_signs)
                add_to_avg_mask = fallback_mask & sign_matches_consensus
                
                sum_original_weights_for_fallback = torch.where(add_to_avg_mask, sum_original_weights_for_fallback + models[model_idx].to(device), sum_original_weights_for_fallback)
                count_for_fallback_avg = torch.where(add_to_avg_mask, count_for_fallback_avg + 1, count_for_fallback_avg)
            
            average_original_weights = torch.where(
                fallback_mask & (count_for_fallback_avg > 0), 
                sum_original_weights_for_fallback / count_for_fallback_avg, 
                base
            )
            
        else:
            sum_original_weights = base.clone()
            for model in models:
                sum_original_weights += model.to(device)
            average_original_weights = sum_original_weights / (1 + num_models)

        fallback_delta = average_original_weights - base
        final_delta_result = torch.where(fallback_mask, fallback_delta, final_delta_result)

    return final_delta_result
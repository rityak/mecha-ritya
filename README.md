# Mecha-Ritya

Extension for ComfyUI Mecha providing additional model merging methods for machine learning.

## Description

Mecha-Ritya contains a set of specialized model merging algorithms for machine learning models.

### Available Merge Methods

#### DELLA Methods

- **della_linear_with_json**: Applies MAGPRUNE (magnitude-based parameter removal) to task vectors and performs weighted linear merging of pruned vectors.
- **della_with_json**: Combines MAGPRUNE with TIES sign election and disjoint merge to resolve sign conflicts during merging.

#### TIES Methods

- **ties_merging_with_json**: Performs TRIM (magnitude-based trimming), ELECT SIGN (sign election), and DISJOINT MERGE (merging only matching signs) for task vectors.
- **ties_lora_merging_with_json**: Applies TIES-MERGING algorithm to pre-computed LoRA deltas with global parameters.

#### Karcher Mean Methods

- **karcher_mean**: Computes Riemannian mean (Karcher mean) on the unit sphere using iterative Weiszfeld algorithm.
- **karcher_mean_with_json**: Karcher mean with parameter configuration via JSON for different model blocks.
- **karcher_mean_with_blocks**: Karcher mean with explicit weight specification for various blocks (input_blocks, middle_block, output_blocks, clip_l, clip_g, etc.).

#### Linear Methods

- **weight_sum_with_json**: Computes normalized weighted sum of tensors, where weights are automatically normalized so their sum equals 1.
- **tensor_sum_with_json**: Computes non-normalized weighted sum of tensors, allowing operations like A + α·(B - C) without constraints on weight sum.

#### Special Methods

- **add_perpendicular**: Adds perpendicular component of vector difference (B - C) to base model A relative to common base C.

### File Structure

```
merge/
├── __init__.py
├── della_json.py                 # DELLA merging methods
├── ties_json.py                   # TIES merging for models
├── ties_lora_json.py              # TIES merging for LoRA
├── karcher.py                     # Karcher Mean
├── karcher_json.py                # Karcher Mean with JSON config
├── karcher_opts.py                # Karcher Mean with block options
├── weight_json.py                 # Normalized weighted sum
├── tensor_sum_json.py             # Non-normalized tensor sum
└── perpendicular.py               # Perpendicular Merge
```

## Installation

### Prerequisites

- ComfyUI
- ComfyUI Mecha (comfy-mecha)

### Installation Instructions

1. **Clone the mecha-ritya repository:**

   ```bash
   git clone https://github.com/rityak/mecha-ritya.git
   ```

2. **Move the cloned folder to the correct location:**

   Copy the `mecha-ritya` folder to the directory:

   ```
   ComfyUI\custom_nodes\comfy-mecha\mecha_extensions\
   ```

3. **The final path should be:**

   ```
   ComfyUI\custom_nodes\comfy-mecha\mecha_extensions\mecha-ritya
   ```

4. **Restart ComfyUI** to load the new merging methods.

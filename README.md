# Mecha-Ritya

Extension for ComfyUI Mecha providing additional model merging methods for machine learning.

## Description

Mecha-Ritya contains a set of specialized model merging algorithms, including:

- **Smooth Peak Merge** - Smooth merging with peak suppression
- **Task Arithmetic** - Task arithmetic for model merging
- **Karcher Mean** - Geometric mean in model space
- **Perpendicular Merge** - Perpendicular merging
- **Multi SLERP** - Spherical linear interpolation
- **Hierarchical Iterative TIES** - Hierarchical iterative merging

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

### File Structure

```
mecha-ritya/
├── __init__.py
├── merge_smooth_peak.py          # Smooth Peak Merge
├── merge_task_arithmetic.py      # Task Arithmetic
├── merge_karcher.py              # Karcher Mean
├── merge_karcher_json.py         # Karcher Mean with JSON config
├── merge_karcher_opts.py         # Karcher Mean with options
├── merge_perpendicular.py        # Perpendicular Merge
├── merge_milti_slerp.py          # Multi SLERP
└── merge_hierarchical_iterative_ties.py  # Hierarchical Iterative TIES
```

## Usage

After installation, the new merging methods will be available in ComfyUI Mecha through the corresponding nodes. Each method has its own parameters and settings that can be configured in the ComfyUI interface.

## Support

For questions and support, refer to the ComfyUI Mecha documentation or create issues in the repository.

# Mecha-Ritya

Extension for ComfyUI Mecha providing additional model merging methods for machine learning.

## Description

Mecha-Ritya contains a set of specialized model merging algorithms, including:

- **Karcher Mean** - Geometric mean in model space
- **Perpendicular Merge** - Perpendicular merging

### File Structure

```
merge/
├── __init__.py
├── merge_karcher.py              # Karcher Mean
├── merge_karcher_json.py         # Karcher Mean with JSON config
├── merge_karcher_opts.py         # Karcher Mean with options
└── merge_perpendicular.py        # Perpendicular Merge
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

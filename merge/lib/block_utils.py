"""
Utilities for determining block names from model keys.
"""
from typing import Dict, Any


def get_user_block_name(key: str, params: Dict[str, Any]) -> str:
    """
    Dynamically determines the user-friendly block name based on the model key.
    This function prioritizes specific block names from the JSON, then group names, then falls back to 'global'.
    
    Args:
        key: Model key (e.g., "model.diffusion_model.input_blocks.0.0.in_layers.0.weight")
        params: Dictionary of parameters from JSON config
    
    Returns:
        Block name string (e.g., "IN00", "IN", "OUT", "global", etc.)
    """
    # 1. Check for specific block number names (e.g., IN00)
    parts = key.split('.')
    if "input_blocks" in parts:
        try:
            index = parts[parts.index("input_blocks") + 1]
            block_name = f"IN{int(index):02}"
            if block_name in params:
                return block_name
        except (ValueError, IndexError):
            pass  # Fallback to group name
    elif "middle_block" in parts:
        if "M00" in params:
            return "M00"
    elif "output_blocks" in parts:
        try:
            index = parts[parts.index("output_blocks") + 1]
            block_name = f"OUT{int(index):02}"
            if block_name in params:
                return block_name
        except (ValueError, IndexError):
            pass  # Fallback to group name

    # 2. Check for broader group names (e.g., IN, OUT)
    if "input_blocks" in parts:
        if "IN" in params:
            return "IN"
    elif "middle_block" in parts:
        if "M" in params:
            return "M"
    elif "output_blocks" in parts:
        if "OUT" in params:
            return "OUT"
    elif key.startswith("conditioner.embedders.0"):
        if "CLIP_L" in params:
            return "CLIP_L"
    elif key.startswith("conditioner.embedders.1"):
        if "CLIP_G" in params:
            return "CLIP_G"
    elif key.startswith("model.diffusion_model.time_embed"):
        if "time_embed" in params:
            return "time_embed"
    elif key.startswith("model.diffusion_model.label_emb"):
        if "label_emb" in params:
            return "label_emb"

    # 3. Fallback to global
    return "global"


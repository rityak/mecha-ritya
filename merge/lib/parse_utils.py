"""
Utilities for parsing weights and JSON parameters.
"""
import json
from typing import Dict, Any, List


def parse_alphas(alphas_str: str) -> List[float]:
    """
    Parses a string of alphas into a list of numbers.
    
    Args:
        alphas_str: Comma-separated string of numbers (e.g., "0.5, 0.3, 0.2")
    
    Returns:
        List of float values
    
    Raises:
        ValueError: If the format is invalid
    """
    try:
        return [float(a.strip()) for a in alphas_str.split(',')]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid weight format: '{alphas_str}'. Weights must be numbers separated by commas.")


def parse_json_params(json_params: str) -> Dict[str, Any]:
    """
    Parses a JSON string into a dictionary with error handling.
    
    Args:
        json_params: JSON string to parse
    
    Returns:
        Dictionary parsed from JSON, or empty dict if string is empty
    
    Raises:
        ValueError: If JSON format is invalid
    """
    if not json_params or not json_params.strip():
        return {}
    
    try:
        return json.loads(json_params)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")


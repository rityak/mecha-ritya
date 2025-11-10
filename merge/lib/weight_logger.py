"""
Utility class for logging weights with deduplication.
"""
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class WeightLogger:
    """
    Logger for weights that avoids redundant logging by tracking last logged values per block.
    """
    
    def __init__(self):
        """Initialize the logger with an empty cache."""
        self._last_logged = {}
    
    def log_weights(self, block_name: str, weights_str: str, method_name: str = ""):
        """
        Log weights for a block if they have changed since last log.
        
        Args:
            block_name: Name of the block (e.g., "IN00", "global")
            weights_str: String representation of weights
            method_name: Optional method name for more descriptive logging
        """
        cache_key = f"{block_name}:{weights_str}"
        if self._last_logged.get(block_name) != weights_str:
            method_prefix = f"[{method_name}] " if method_name else ""
            logging.info(f"{method_prefix}Using weights '{weights_str}' for block '{block_name}'.")
            self._last_logged[block_name] = weights_str


"""
Tokenization utilities for converting between wave values and ECG tokens.

Uses the same binning scheme as ecg_raw_dataloader:
- 600 bins from -3 to 3 mV
- Token format: ecg_{bin_index}
"""

import numpy as np
from typing import List, Tuple

from ecg_bench.configs.constants import (
    ECG_RAW_TOKEN_PREFIX,
    ECG_RAW_NUM_BINS,
    ECG_RAW_MIN_VALUE,
    ECG_RAW_MAX_VALUE,
)


def value_to_bin(value: float) -> int:
    """
    Convert a single value to its bin index.

    Args:
        value: Value in mV (will be clamped to [-3, 3])

    Returns:
        Bin index (0-599)
    """
    clamped = np.clip(value, ECG_RAW_MIN_VALUE, ECG_RAW_MAX_VALUE)
    bin_width = (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS
    bin_idx = int((clamped - ECG_RAW_MIN_VALUE) / bin_width)
    return np.clip(bin_idx, 0, ECG_RAW_NUM_BINS - 1)


def bin_to_value(bin_idx: int) -> float:
    """
    Convert a bin index to its center value.

    Args:
        bin_idx: Bin index (0-599)

    Returns:
        Center value of the bin in mV
    """
    bin_width = (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS
    return ECG_RAW_MIN_VALUE + (bin_idx + 0.5) * bin_width


def values_to_tokens(values: np.ndarray) -> List[str]:
    """
    Convert array of values to token strings.

    Args:
        values: Array of values in mV

    Returns:
        List of token strings (e.g., ["ecg_300", "ecg_301", ...])
    """
    clamped = np.clip(values, ECG_RAW_MIN_VALUE, ECG_RAW_MAX_VALUE)
    bin_width = (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS
    bin_indices = ((clamped - ECG_RAW_MIN_VALUE) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, ECG_RAW_NUM_BINS - 1)
    return [f"{ECG_RAW_TOKEN_PREFIX}{idx}" for idx in bin_indices]


def tokens_to_values(tokens: List[str]) -> np.ndarray:
    """
    Convert token strings to center values.

    Args:
        tokens: List of token strings (e.g., ["ecg_300", "ecg_301", ...])

    Returns:
        Array of center values in mV
    """
    bin_indices = [int(tok.replace(ECG_RAW_TOKEN_PREFIX, "")) for tok in tokens]
    return np.array([bin_to_value(idx) for idx in bin_indices])


def value_to_token(value: float) -> str:
    """Convert a single value to a token string."""
    return f"{ECG_RAW_TOKEN_PREFIX}{value_to_bin(value)}"


def token_to_value(token: str) -> float:
    """Convert a single token string to its center value."""
    bin_idx = int(token.replace(ECG_RAW_TOKEN_PREFIX, ""))
    return bin_to_value(bin_idx)


def format_value(value: float, precision: int = 2) -> str:
    """Format a value for display in prompts."""
    return f"{value:.{precision}f}"


def format_values_list(values: np.ndarray, precision: int = 2) -> str:
    """Format a list of values as comma-separated string."""
    return ", ".join([format_value(v, precision) for v in values])


def tokens_to_string(tokens: List[str]) -> str:
    """Convert list of tokens to concatenated string (no spaces)."""
    return "".join(tokens)


def string_to_tokens(token_string: str) -> List[str]:
    """Convert concatenated token string to list.

    Handles both formats:
    - Concatenated: "ecg_300ecg_301ecg_302"
    - Space-separated: "ecg_300 ecg_301 ecg_302"
    """
    token_string = token_string.strip()
    if " " in token_string:
        return token_string.split()
    # Parse concatenated format: split on 'ecg_' and reconstruct
    import re
    matches = re.findall(r'ecg_\d+', token_string)
    return matches if matches else []


def get_bin_range() -> Tuple[int, int]:
    """Get the valid bin index range."""
    return 0, ECG_RAW_NUM_BINS - 1


def get_value_range() -> Tuple[float, float]:
    """Get the valid value range."""
    return ECG_RAW_MIN_VALUE, ECG_RAW_MAX_VALUE


def get_bin_width() -> float:
    """Get the width of each bin."""
    return (ECG_RAW_MAX_VALUE - ECG_RAW_MIN_VALUE) / ECG_RAW_NUM_BINS


def compare_magnitudes(token1: str, token2: str) -> int:
    """
    Compare magnitudes of two tokens.

    Returns:
        1 if token1 > token2
        -1 if token1 < token2
        0 if equal
    """
    val1 = abs(token_to_value(token1))
    val2 = abs(token_to_value(token2))
    if val1 > val2:
        return 1
    elif val1 < val2:
        return -1
    return 0


def get_halfway_token(token1: str, token2: str) -> str:
    """Get the token that represents the halfway point between two tokens."""
    val1 = token_to_value(token1)
    val2 = token_to_value(token2)
    halfway_val = (val1 + val2) / 2
    return value_to_token(halfway_val)


def get_next_token(token: str, step: int = 1) -> str:
    """Get the token that is step bins away from the given token."""
    bin_idx = int(token.replace(ECG_RAW_TOKEN_PREFIX, ""))
    new_idx = np.clip(bin_idx + step, 0, ECG_RAW_NUM_BINS - 1)
    return f"{ECG_RAW_TOKEN_PREFIX}{new_idx}"

"""Argument checks shared by the chunking functions."""

import numpy as np


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)):
        raise TypeError(f"chunk_size must be an int, got {type(chunk_size).__name__}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

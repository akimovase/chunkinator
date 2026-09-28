"""Split a DataFrame into chunks without splitting groups of equal values."""

from chunkinator.contiguous import chunk_bounds, iter_chunks

__all__ = ["chunk_bounds", "iter_chunks"]

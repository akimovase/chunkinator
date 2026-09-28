"""Split a DataFrame into chunks without splitting groups of equal values."""

from chunkinator.contiguous import chunk_bounds, iter_chunks
from chunkinator.grouped import iter_grouped_chunks

__all__ = ["chunk_bounds", "iter_chunks", "iter_grouped_chunks"]

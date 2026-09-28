"""Chunking algorithm for the case when we don't
want intersection odf date intervals between chunks."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from chunkinator._validation import _validate_chunk_size

__all__ = ["chunk_bounds", "iter_chunks"]


def chunk_bounds(values: npt.NDArray[Any], chunk_size: int) -> Iterator[tuple[int, int]]:
    """Generating half-open bounds of chunks over sorted ``values``.

    Every chunk except possibly the last one has at least ``chunk_size`` elements,
    equal values never end up in different chunks.
    """
    _validate_chunk_size(chunk_size)
    return _generate_bounds(values, chunk_size)


def _generate_bounds(values: npt.NDArray[Any], chunk_size: int) -> Iterator[tuple[int, int]]:
    n = len(values)
    start = 0
    while start < n:
        last = start + chunk_size - 1
        end = n if last >= n - 1 else int(np.searchsorted(values, values[last], side="right"))
        yield start, end
        start = end


def _key_values(series: pd.Series[Any]) -> npt.NDArray[Any]:
    """Return the series values as a numpy array"""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return np.asarray(series.array.asi8)
    return series.to_numpy()


def iter_chunks(
    df: pd.DataFrame, column: str, chunk_size: int, *, sort: bool = False
) -> Iterator[pd.DataFrame]:
    """Spliting ``df`` into chunks by the values of ``column``,
      chunks date intervals don't intersect.

    Each chunk has at least ``chunk_size`` rows (only the last one may be
    smaller), and rows with equal ``column`` values always land in the same
    chunk.

    The chunks are iloc slices of ``df`` and do not copy its data.
    """
    _validate_chunk_size(chunk_size)
    if column not in df.columns:
        raise KeyError(column)
    series = df[column]
    if not series.is_monotonic_increasing:
        if series.hasnans:
            raise ValueError(f"column {column!r} contains missing values")
        if not sort:
            raise ValueError(
                f"column {column!r} is not sorted ascending; pass sort=True to sort it"
            )
        df = df.sort_values(column, kind="stable")
        series = df[column]

    bounds = _generate_bounds(_key_values(series), chunk_size)
    return (df.iloc[start:end] for start, end in bounds)

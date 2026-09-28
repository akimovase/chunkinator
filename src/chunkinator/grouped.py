"""Chunking algorithm for the case when we can have
intersection of date intervals between chunks."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from chunkinator._validation import _validate_chunk_size

__all__ = ["iter_grouped_chunks"]


def _cover_groups(sizes: list[int], chunk_size: int) -> tuple[npt.NDArray[np.intp], int]:
    """Assign groups of dates to chunks, return the chunk of each group and the short chunk.

    Every chunk except at most one gets ``chunk_size`` rows or more. The id of
    that short chunk is returned, or -1 if all chunks are full.
    """
    chunk_of = np.empty(len(sizes), dtype=np.intp)
    n_chunks = 0
    groups_by_size: dict[int, list[int]] = {}
    for group in range(len(sizes) - 1, -1, -1):
        size = sizes[group]
        if size >= chunk_size:
            chunk_of[group] = n_chunks
            n_chunks += 1
        else:
            groups_by_size.setdefault(size, []).append(group)
    available = sorted(groups_by_size)

    def take_needed_group(pos: int) -> tuple[int, int]:
        size = available[pos]
        same_size = groups_by_size[size]
        group = same_size.pop()
        if not same_size:
            del groups_by_size[size]
            available.pop(pos)
        return group, size

    level = chunk_size
    while available:
        group, level = take_needed_group(len(available) - 1)
        chunk_of[group] = n_chunks
        while level < chunk_size and available:
            pos = bisect_left(available, chunk_size - level)
            group, size = take_needed_group(min(pos, len(available) - 1))
            chunk_of[group] = n_chunks
            level += size
        n_chunks += 1
    short = n_chunks - 1 if level < chunk_size else -1

    return chunk_of, short


def _order_chunks(chunk_of: npt.NDArray[np.intp], short: int) -> npt.NDArray[np.intp]:
    """Renumber chunks by their smallest value and move the short chunk to the end."""
    renumbered, old_ids = pd.factorize(chunk_of)
    if short >= 0:
        short_new = int(np.flatnonzero(old_ids == short)[0])
        is_short = renumbered == short_new
        renumbered[renumbered > short_new] -= 1
        renumbered[is_short] = len(old_ids) - 1
    return renumbered


def iter_grouped_chunks(df: pd.DataFrame, column: str, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Spliting ``df`` into chunks by the values of ``column``, chunks date intervals can intersect.

    Rows with equal ``column`` values always land in the same chunk. Every chunk
    except the last one has at least ``chunk_size`` rows, and a group of
    ``chunk_size`` rows or more forms a chunk of its own.
    Chunks are not contiguous ranges of ``column``: full chunks are ordered by their smallest
    ``column`` value, the short one comes last, and rows inside a chunk
    keep the order they had in ``df``.
    """
    _validate_chunk_size(chunk_size)
    if column not in df.columns:
        raise KeyError(column)
    series = df[column]
    if series.hasnans:
        raise ValueError(f"column {column!r} contains missing values")

    codes, _ = pd.factorize(series, sort=True)
    chunk_of_group, short = _cover_groups(np.bincount(codes).tolist(), chunk_size)
    chunk_of_row = _order_chunks(chunk_of_group, short)[codes]
    del codes
    row_order = np.argsort(chunk_of_row, kind="stable")
    ends = np.cumsum(np.bincount(chunk_of_row)).tolist()
    return _take_chunks(df, row_order, ends)


def _take_chunks(
    df: pd.DataFrame, row_order: npt.NDArray[Any], ends: list[int]
) -> Iterator[pd.DataFrame]:
    start = 0
    for end in ends:
        yield df.take(row_order[start:end])
        start = end

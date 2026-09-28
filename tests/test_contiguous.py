from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from chunkinator import chunk_bounds, iter_chunks
from helpers import chunk_values, make_frame


@pytest.mark.parametrize(
    ("chunk_size", "expected"),
    [
        (1, [[1, 1], [2, 2, 2], [3]]),
        (2, [[1, 1], [2, 2, 2], [3]]),
        (3, [[1, 1, 2, 2, 2], [3]]),
        (4, [[1, 1, 2, 2, 2], [3]]),
        (5, [[1, 1, 2, 2, 2], [3]]),
        (6, [[1, 1, 2, 2, 2, 3]]),
        (100, [[1, 1, 2, 2, 2, 3]]),
    ],
)
def test_task_example(example_df: pd.DataFrame, chunk_size: int, expected: list[list[int]]) -> None:
    assert chunk_values(list(iter_chunks(example_df, "dt", chunk_size))) == expected


def test_task_dataframe() -> None:
    dfs = pd.date_range("2023-01-01 00:00:00", "2023-01-01 00:00:05", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})

    assert [len(c) for c in iter_chunks(df, "dt", 1)] == [3] * 6
    assert [len(c) for c in iter_chunks(df, "dt", 4)] == [6, 6, 6]
    assert [len(c) for c in iter_chunks(df, "dt", 7)] == [9, 9]
    assert [len(c) for c in iter_chunks(df, "dt", 10)] == [12, 6]
    assert [len(c) for c in iter_chunks(df, "dt", 18)] == [18]


def test_empty_frame() -> None:
    df = pd.DataFrame({"dt": pd.Series([], dtype="datetime64[ns]")})
    assert list(iter_chunks(df, "dt", 3)) == []


def test_single_row() -> None:
    chunks = list(iter_chunks(make_frame([0]), "dt", 5))
    assert chunk_values(chunks) == [[0]]


def test_group_larger_than_chunk_size_is_not_split() -> None:
    chunks = list(iter_chunks(make_frame([7] * 10), "dt", 3))
    assert chunk_values(chunks) == [[7] * 10]


def test_all_unique_values() -> None:
    chunks = list(iter_chunks(make_frame(list(range(7))), "dt", 3))
    assert [len(c) for c in chunks] == [3, 3, 1]


def test_other_columns_and_index_are_preserved() -> None:
    df = make_frame([1, 1, 2, 3, 3]).set_index(pd.Index(list("abcde")))
    chunks = list(iter_chunks(df, "dt", 2))
    pd.testing.assert_frame_equal(pd.concat(chunks), df)
    assert list(chunks[0].index) == ["a", "b"]


def test_chunks_are_views() -> None:
    df = make_frame([1, 1, 2, 2, 3])
    source = df["payload"].to_numpy()
    for chunk in iter_chunks(df, "dt", 2):
        assert np.shares_memory(chunk["payload"].to_numpy(), source)


def test_modifying_chunk_does_not_change_source() -> None:
    df = make_frame([1, 1, 2, 2, 3])
    original = df.copy()
    for chunk in iter_chunks(df, "dt", 2):
        chunk["payload"] = -1
    pd.testing.assert_frame_equal(df, original)


def test_timezone_aware_column() -> None:
    df = make_frame([1, 1, 2, 3])
    df["dt"] = df["dt"].dt.tz_localize("Europe/Moscow")
    assert [len(c) for c in iter_chunks(df, "dt", 1)] == [2, 1, 1]


def test_non_datetime_column() -> None:
    df = pd.DataFrame({"k": [1, 1, 2, 2, 2, 3]})
    assert [len(c) for c in iter_chunks(df, "k", 3)] == [5, 1]


def test_unsorted_raises_by_default() -> None:
    with pytest.raises(ValueError, match="sort=True"):
        iter_chunks(make_frame([2, 1, 2]), "dt", 1)


def test_unsorted_with_sort() -> None:
    df = make_frame([2, 1, 2, 1, 3])
    chunks = list(iter_chunks(df, "dt", 1, sort=True))
    assert chunk_values(chunks) == [[1, 1], [2, 2], [3]]
    assert list(chunks[0]["payload"]) == [1, 3]
    assert list(chunks[1]["payload"]) == [0, 2]


def test_missing_values_raise() -> None:
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01", "NaT"])})
    with pytest.raises(ValueError, match="missing"):
        iter_chunks(df, "dt", 1)


def test_unknown_column_raises(example_df: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        iter_chunks(example_df, "nope", 1)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_raises(example_df: pd.DataFrame, chunk_size: int) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        iter_chunks(example_df, "dt", chunk_size)


@pytest.mark.parametrize("chunk_size", [1.5, "2", None, True])
def test_non_int_chunk_size_raises(example_df: pd.DataFrame, chunk_size: object) -> None:
    with pytest.raises(TypeError):
        iter_chunks(example_df, "dt", chunk_size)  # type: ignore[arg-type]


def test_numpy_int_chunk_size(example_df: pd.DataFrame) -> None:
    assert len(list(iter_chunks(example_df, "dt", np.int64(3)))) == 2  # type: ignore[arg-type]


def test_chunk_bounds_on_numpy_array() -> None:
    values = np.array([1, 1, 2, 2, 2, 3])
    assert list(chunk_bounds(values, 3)) == [(0, 5), (5, 6)]


def test_chunk_bounds_validates_eagerly() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        chunk_bounds(np.arange(3), 0)


@given(
    counts=st.lists(st.integers(min_value=1, max_value=8), max_size=40),
    chunk_size=st.integers(min_value=1, max_value=30),
)
def test_chunking_invariants(counts: list[int], chunk_size: int) -> None:
    values = np.repeat(np.arange(len(counts)), counts)
    bounds = list(chunk_bounds(values, chunk_size))
    edges = [0] + [end for _, end in bounds]
    assert bounds == list(pairwise(edges))
    assert edges[-1] == len(values)

    for i, (start, end) in enumerate(bounds):
        chunk = values[start:end]
        if start > 0:
            assert values[start - 1] != values[start]
        if i < len(bounds) - 1:
            assert len(chunk) >= chunk_size
        last_group_len = int(np.sum(chunk == chunk[-1]))
        assert len(chunk) - last_group_len < chunk_size

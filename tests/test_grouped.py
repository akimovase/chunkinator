import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from chunkinator import iter_grouped_chunks
from helpers import chunk_values, make_frame


def _repeat(sizes: list[int]) -> list[int]:
    """Group sizes -> seconds: [3, 1] -> [1, 1, 1, 2]."""
    return [second for second, size in enumerate(sizes, start=1) for _ in range(size)]


def test_small_groups_are_combined_up_to_chunk_size() -> None:
    dfs = pd.date_range("2023-01-01 00:00:01", "2023-01-01 00:00:04", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    assert [len(c) for c in iter_grouped_chunks(df, "dt", 4)] == [6, 6]


def test_groups_are_combined_into_exact_chunks() -> None:
    df = make_frame(_repeat([3, 4, 2, 1]))
    chunks = list(iter_grouped_chunks(df, "dt", 5))
    assert chunk_values(chunks) == [[1, 1, 1, 3, 3], [2, 2, 2, 2, 4]]
    assert list(chunks[0]["payload"]) == [0, 1, 2, 7, 8]


def test_largest_group_is_closed_by_smallest_fitting_one() -> None:
    df = make_frame(_repeat([4, 2, 1, 3, 5, 6]))
    chunks = list(iter_grouped_chunks(df, "dt", 7))
    assert [sorted(set(v)) for v in chunk_values(chunks)] == [[1, 4], [2, 5], [3, 6]]
    assert [len(c) for c in chunks] == [7, 7, 7]


@pytest.mark.parametrize(
    ("chunk_size", "expected"),
    [
        (1, [[1, 1], [2, 2, 2], [3]]),
        (2, [[1, 1], [2, 2, 2], [3]]),
        (3, [[1, 1, 3], [2, 2, 2]]),
        (4, [[2, 2, 2, 3], [1, 1]]),
        (5, [[1, 1, 2, 2, 2], [3]]),
        (6, [[1, 1, 2, 2, 2, 3]]),
    ],
)
def test_task_example(example_df: pd.DataFrame, chunk_size: int, expected: list[list[int]]) -> None:
    assert chunk_values(list(iter_grouped_chunks(example_df, "dt", chunk_size))) == expected


def test_group_of_chunk_size_or_more_is_alone() -> None:
    df = make_frame(_repeat([1, 5, 1, 3]))
    assert chunk_values(list(iter_grouped_chunks(df, "dt", 3))) == [[2] * 5, [4] * 3, [1, 3]]


def test_short_chunk_comes_last() -> None:
    df = make_frame(_repeat([1, 4, 4]))
    assert [len(c) for c in iter_grouped_chunks(df, "dt", 4)] == [4, 4, 1]


def test_empty_frame() -> None:
    df = pd.DataFrame({"dt": pd.Series([], dtype="datetime64[ns]")})
    assert list(iter_grouped_chunks(df, "dt", 3)) == []


def test_unsorted_input() -> None:
    df = make_frame([2, 1, 2, 3, 1, 2])
    chunks = list(iter_grouped_chunks(df, "dt", 3))
    assert [sorted(set(v)) for v in chunk_values(chunks)] == [[1, 3], [2]]
    assert list(chunks[0]["payload"]) == [1, 3, 4]


def test_modifying_chunk_does_not_change_source() -> None:
    df = make_frame([1, 1, 2])
    original = df.copy()
    for chunk in iter_grouped_chunks(df, "dt", 2):
        chunk["payload"] = -1
    pd.testing.assert_frame_equal(df, original)


def test_missing_values_raise() -> None:
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01", "NaT"])})
    with pytest.raises(ValueError, match="missing"):
        iter_grouped_chunks(df, "dt", 1)


def test_unknown_column_raises() -> None:
    with pytest.raises(KeyError):
        iter_grouped_chunks(make_frame([1]), "nope", 1)


@pytest.mark.parametrize(
    ("chunk_size", "error"), [(0, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_bad_chunk_size_raises(chunk_size: object, error: type[Exception]) -> None:
    with pytest.raises(error, match="chunk_size"):
        iter_grouped_chunks(make_frame([1]), "dt", chunk_size)  # type: ignore[arg-type]


@given(
    seconds=st.lists(st.integers(min_value=0, max_value=15), max_size=60),
    chunk_size=st.integers(min_value=1, max_value=12),
)
def test_grouped_invariants(seconds: list[int], chunk_size: int) -> None:
    df = make_frame(seconds)
    chunks = list(iter_grouped_chunks(df, "dt", chunk_size))
    group_size = df["dt"].value_counts()

    pd.testing.assert_frame_equal(pd.concat([df.iloc[:0], *chunks]).sort_index(), df)
    seen: set[pd.Timestamp] = set()
    for i, chunk in enumerate(chunks):
        values = set(chunk["dt"])
        assert not values & seen
        seen |= values
        assert chunk.index.is_monotonic_increasing
        if i < len(chunks) - 1:
            assert len(chunk) >= chunk_size
        if len(values) > 1:
            sizes = [group_size[v] for v in values]
            assert max(sizes) < chunk_size
            assert len(chunk) - min(sizes) < chunk_size

    full_firsts = [chunk["dt"].min() for chunk in chunks if len(chunk) >= chunk_size]
    assert full_firsts == sorted(full_firsts)

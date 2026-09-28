import pandas as pd

BASE = pd.Timestamp("2023-01-01")


def make_frame(seconds: list[int]) -> pd.DataFrame:
    """List of seconds -> Dataframe with ``dt`` column (``BASE`` + given seconds)
    and ``payload`` (holds the original row numbers)"""
    dt = BASE + pd.to_timedelta(seconds, unit="s")
    return pd.DataFrame({"dt": dt, "payload": range(len(seconds))})


def chunk_values(chunks: list[pd.DataFrame]) -> list[list[int]]:
    """Chunks -> lists of seconds"""
    return [[int((ts - BASE).total_seconds()) for ts in chunk["dt"]] for chunk in chunks]

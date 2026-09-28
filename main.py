import pandas as pd

from chunkinator import iter_chunks, iter_grouped_chunks


def main() -> None:
    dfs = pd.date_range("2023-01-01 00:00:01", "2023-01-01 00:00:04", freq="s")
    df = pd.DataFrame({"dt": dfs.repeat(3)})
    for chunk in iter_chunks(df, "dt", 4):
        print(chunk, "\n")
    print("end")
    for chunk in iter_grouped_chunks(df, "dt", 4):
        print(chunk, "\n")


if __name__ == "__main__":
    main()

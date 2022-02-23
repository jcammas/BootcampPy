import pandas as pd


class FileLoader():
    @staticmethod
    def load(path: str, header=True) -> pd.DataFrame:
        """takes as an argument the file path of the dataset to load, displays a
        message specifying the dimensions of the dataset (e.g. 340 x 500) and returns the
        dataset loaded as a pandas.DataFrame"""
        header = ([0] if header else None)
        ret = pd.read_csv(path, header=header)
        print(f"Loading dataset of dims {ret.shape[0]} x {ret.shape[1]}")
        return ret

    @staticmethod
    def display(df: pd.DataFrame, n: int) -> None:
        """takes a pandas.DataFrame and an integer as arguments, displays
        the first n rows of the dataset if n is positive, or the last n rows if n is negative."""
        try:
            tmp = df[:n] if (int(n) >= 0) else df[n:]
            print(tmp)
        except ValueError as err:
            pass


if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("../resources/athlete_events.csv")
    loader.display(data, 3)
    loader.display(data, -3)
    loader.display(data, 0)
    loader.display(data, "lol")
    # loader.display(data, -12)

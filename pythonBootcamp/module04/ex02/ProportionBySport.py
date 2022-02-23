import pandas as pd
from FileLoader import FileLoader


def proportionBySport(df: pd.DataFrame, year: int, sport: str, gender: str) -> None:
    df = df[(df["Year"] == year) & (df["Sex"] == gender)]
    df_Sport = df[df["Sport"] == sport]
    return ((df_Sport.shape[0] / df.shape[0]) * 100)


if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("../resources/athlete_events.csv")

    print("")
    print(proportionBySport(data, 2004, 'Tennis', 'F'), end="\n\n")
    print(proportionBySport(data, 2008, 'Hockey', 'F'), end="\n\n")
    print(proportionBySport(data, 1964, 'Biathlon', 'M'), end="\n\n")

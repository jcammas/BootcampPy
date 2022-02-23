import pandas as pd
from FileLoader import FileLoader


class SpatioTemporalData():
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = data.loc[:, ['Year', 'City']].drop_duplicates()

    def when(self, location: str) -> list:
        return list(self.df[self.df.City == location]['Year'])

    def where(self, date: int) -> list:
        return list(self.df[self.df.Year == date]['City'])


if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load('../resources/athlete_events.csv')

    sp = SpatioTemporalData(data)
    print(sp.where(2000))
    print(sp.where(1980))
    print(sp.when('London'))

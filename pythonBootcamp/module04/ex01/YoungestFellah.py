import pandas as pd
from FileLoader import FileLoader


def youngestfellah(data, val):
    print("1", data.head())
    y = data[data.Year == val]
    print("2", y.head())
    print("3", data.head())
    return {
        'f': y[y.Sex == 'F'].Age.min(),
        'm': y[y.Sex == 'M'].Age.min(),

    }


if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("../resources/athlete_events.csv")
    print(youngestfellah(data, 1992))
    print("-------------------------")
    print(youngestfellah(data, 2004))
    print("-------------------------")
    print(youngestfellah(data, 2010))
    print("-------------------------")
    print(youngestfellah(data, 2003))

from unittest.util import sorted_list_difference
import pandas as pd
from FileLoader import FileLoader


def howManyMedalsByCountry(df, country):
    tmp = dict()
    df = df[df.Team == country].copy()

    for i, c in df.iterrows():
        y = c["Year"]
        m = c["Medal"]
        if y not in tmp.keys():
            tmp.update({y: {"G": 0, "S": 0, "B": 0}})
        if m == "Bronze":
            tmp[y]["B"] = tmp[y]["B"] + 1
        elif m == "Silver":
            tmp[y]["S"] = tmp[y]["S"] + 1
        elif m == "Gold":
            tmp[y]["G"] = tmp[y]["G"] + 1

    res = sorted(tmp.items(), key=lambda x: x[0], reverse=False)

    return (res)


if __name__ == "__main__":
    loader = FileLoader()
    df = loader.load("../resources/athlete_events.csv")
    print(howManyMedalsByCountry(df, "United States"))

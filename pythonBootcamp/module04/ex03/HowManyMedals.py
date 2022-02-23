import pandas as pd
from FileLoader import FileLoader


def howManyMedals(df, name):
    tmp = dict()
    df = df[df.Name == name].copy()
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
    return (tmp)


if __name__ == "__main__":
    loader = FileLoader()
    data = loader.load("../resources/athlete_events.csv")

    print(howManyMedals(data, 'Gary Abraham'))
    print("")
    print(howManyMedals(data, 'Yekaterina Konstantinovna Abramova'))
    print("")
    print(howManyMedals(data, 'Kristin Otto'))

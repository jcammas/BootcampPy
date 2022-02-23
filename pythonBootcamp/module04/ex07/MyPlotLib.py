import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from FileLoader import FileLoader


class MyPlotLib:
    def histogram(self, df: pd.DataFrame, features: list):
        df[features].hist()
        plt.show()

    def density(self, df: pd.DataFrame, features: list):
        df[features].plot.kde()
        plt.show()

    def pair_plot(self, df: pd.DataFrame, features: list):
        sns.pairplot(df[features], markers=".", height=2,
                     plot_kws=dict(linewidth=0))
        plt.show()

    def box_plot(self, df: pd.DataFrame, features: list):
        sns.boxplot(data=df[features])
        plt.show()


loader = FileLoader()
data = loader.load('../resources/athlete_events.csv')
pl = MyPlotLib()
pl.histogram(data, ["Height", "Weight"])
pl.density(data, ["Weight", "Height"])
pl.pair_plot(data, ["Weight", "Height"])
pl.box_plot(data, ["Weight", "Height"])

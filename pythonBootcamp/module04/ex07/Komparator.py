import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from FileLoader import FileLoader


class Komparator:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def compare_box_plots(self, categorical_var: str, numerical_var: str):
        '''displays a box plot with
        several boxes to compare how the distribution of the numerical variable changes if
        we only consider the subpopulation which belongs to each category. There should
        be as many boxes as categories. For example, with Sex and Height, we would
        compare the height distributions of men vs. women with two boxes on the same
        graph'''
        nbr = len(self.df[categorical_var].drop_duplicates())
        c, ax = plt.subplots(1, nbr, figsize=(9, 4))
        for i, elem in enumerate(self.df[categorical_var].drop_duplicates()):
            ax[i].set_title(f'{categorical_var}/{elem}')
            sbn.boxplot(x=self.df[numerical_var][self.df[categorical_var] == elem].dropna(
            ), ax=ax[i], color='pink')
        plt.show()

    def density(self, categorical_var: str, numerical_var: str):
        '''displays the density of the numerical variable. Each subpopulation should be represented by a separate curve on the
        graph'''
        plt.figure(figsize=(6, 4))
        for i, elem in enumerate(self.df[categorical_var].drop_duplicates()):
            sbn.kdeplot(
                self.df[numerical_var][self.df[categorical_var] == elem].dropna(), label=elem)
        plt.title(f'{categorical_var}')
        plt.show()

    def compare_histograms(self, categorical_var: str, numerical_var: str):
        '''plots the numerical
        variable in a separate histogram for each category. As an extra, you can use overlapping histograms with a color code (but no extra point will be granted!).'''
        nbr = len(self.df[categorical_var].drop_duplicates())

        c, ax = plt.subplots(1, nbr + 1, figsize=(9, 4))
        for i, elem in enumerate(self.df[categorical_var].drop_duplicates()):
            ax[i].set_title(
                f'{categorical_var}/{elem}')
            sbn.histplot(self.df[numerical_var][self.df[categorical_var] == elem].dropna(
            ), kde=False, ax=ax[i])
        plt.show()


loader = FileLoader()
data = loader.load('../resources/athlete_events.csv')
kp = Komparator(data)
# kp.compare_box_plots("Medal", "Age")
# kp.density("Medal", "Height")
kp.compare_histograms("Medal", "Height")

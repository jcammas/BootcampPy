import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D


class KmeansClustering:
    def __init__(self, max_iter=30, ncentroid=4):
        self.ncentroid = ncentroid
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, X):
        """
        Run the K-means clustering algorithm.
        For the location of the initial centroids, random pick ncentroids from the dataset.
        Args:
        X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Returns:
        None.
        Raises:
        This function should not raise any Exception.
        """
        self.kmeans = KMeans(
            init='random',  n_clusters=self.ncentroid, n_init=self.max_iter)
        self.kmeans.fit(X)

    def predict(self, X):
        """
        Predict from wich cluster each datapoint belongs to.
        Args:
        X: has to be an numpy.ndarray, a matrice of dimension m * n.
        Returns:
        the prediction has a numpy.ndarray, a vector of dimension m * 1.
        Raises:
        This function should not raise any Exception.
        """
        self.p = self.kmeans.predict(X)
        self.centroids = self.kmeans.cluster_centers_
        return self.centroids

    def fig_3D(self, X):
        fig = plt.figure()
        ax = Axes3D(fig)
        # fig.add_axes(ax)

        p = self.p
        c = self.centroids

        ax.set_xlabel("Height")
        ax.set_ylabel("Weight")
        ax.set_zlabel("Bone Density")

        color = ["red", "blue", "green", "pink"]
        for i in range(self.ncentroid):
            ax.scatter(X[p == i, 0], X[p == i, 1],
                       X[p == i, 2], color=color[i])
            ax.scatter(c[i, 0], c[i, 1],
                       c[i, 2], color=color[i], marker="o", s=150, label="centroids")
        plt.show()


if __name__ == "__main__":
    data = np.genfromtxt(
        "../resources/solar_system_census.csv", delimiter=",", skip_header=1)
    X = data[:, 1:]  # Delete index
    ncentroid = 4

    kms = KmeansClustering(ncentroid=ncentroid)
    kms.fit(X)
    kms.predict(X)
    kms.fig_3D(X)

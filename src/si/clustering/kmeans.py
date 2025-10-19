import numpy as np
from si.base.model import Model
from si.base.transformer import Transformer
from si.data.dataset import Dataset, dataset
from si.statistics.euclidean_distance import euclidian_distance


class Kmeans(Transformer, Model):
    def __init__(self, k, max_iter=100, distance=euclidian_distance, **kwargs):
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset):
        random_indexes = np.random.permutation(dataset.shape()[0])
        self.centroids = dataset.X[random_indexes, :]

    def _calculate_distances(self, sample):
        return self.distance(sample, self.centroids)

    def _get_closest_centroid(self, sample):
        centroids_distance = self._calculate_distances(sample)
        centroids_index = np.argmin(centroids_distance, axis=0)
        return centroids_index

    def _fit(self, dataset):
        self._init_centroids(dataset)

        i = 0
        convergence = False
        labels = np.zeros(dataset.shape()[0])

        while not convergence and i < self.max_iter:
            new_labels = np.apply_along_axis(self._get_closest_centroid, arr=dataset.X, axis = 1)


            self.labels = new_labels
            centroids = []
            for j in range(self.k):
                mask = new_labels == j
                new_centroids = np.mean(dataset.X[mask])
                centroids.append(new_centroids)

            centroids = np.array(centroids)
            convergence = not np.any(new_labels != labels)
            labels = new_labels
            i += 1

        self.labels = labels
        return self

    def _transform(self):
        return np.apply_along_axis(self._calculate_distances, arr = dataset.X, axis = 1)

    def _predict(self, dataset):
        new_labels = np.apply_along_axis(self.labels, arr=dataset.X, axis=1)
        return new_labels

from si.base.model import Model
import numpy as np
from si.data.dataset import Dataset


class KNNClassifier(Model):

    def __init__(self, k:int,distance_func:callable, **kwargs):

        self.k = k
        self.distance_func = distance_func
        self.dataset = None



    def _fit(self, datset) -> "KNN":

        self.dataset = datset

    def _get_closest_neighbors(self, sample:np.ndarray) -> np.ndarray:
        distance_to_all_points = self.distance_func(sample, self.dataset.X)
        indexes_of_nn = np.argsort(distance_to_all_points)[:self.k]
        nn_labels = self.dataset.y[indexes_of_nn]
        unique_labels, counts = np.unique(nn_labels, return_counts=True)
        label = unique_labels[np.argmax(counts)]
        return label


    def _predict(self, dataset) -> np.ndarray:

        return np.apply_along_axis(self._get_closest_neighbors(),axis = 1, array=dataset.X)



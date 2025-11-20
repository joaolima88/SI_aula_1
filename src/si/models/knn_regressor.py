from typing import Callable, Union
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    A K-Nearest Neighbors (KNN) regressor that predicts target values
    by averaging the values of the k-nearest samples in the dataset.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _label_average(self, sample: np.ndarray) -> float:
        """
        Calculate the average target value of the k-nearest neighbors.

        Parameters
        ----------
        sample : np.ndarray
            The sample to predict the target value for.

        Returns
        -------
        float
            The predicted target value based on neighbors' average.
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors indexes
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the values of the k nearest neighbors (y values)
        k_nearest_neighbors_values = self.dataset.y[k_nearest_neighbors]

        # get the average of the k nearest neighbors
        k_nearest_neighbors_values_mean = np.mean(k_nearest_neighbors_values)

        return k_nearest_neighbors_values_mean

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the target values of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the target values for

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        # apply_along_axis applies the function to each row (sample) in dataset.X
        predictions = np.apply_along_axis(self._label_average, axis=1, arr=dataset.X)
        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluate the model using Root Mean Squared Error (RMSE).

        Parameters
        ----------
        dataset : Dataset
            The dataset with true target values.
        predictions : np.ndarray
            Predicted target values.

        Returns
        -------
        float
            RMSE value indicating the prediction error.
        """
        return rmse(dataset.y, predictions)
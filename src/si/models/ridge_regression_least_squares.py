import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    The Ridge Regression model using the Least Squares method (closed form solution).
    Linear regression with L2 regularization.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.ndarray
        The model parameters (coefficients). Includes the intercept at index 0.
    mean: np.ndarray
        The mean of the features (used for scaling)
    std: np.ndarray
        The standard deviation of the features (used for scaling)
    """

    def __init__(self, l2_penalty: float = 1, scale: bool = True, **kwargs):
        """
        Initialize the Ridge Regression Least Squares model
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fits the model to the dataset using the closed form solution.

        Parameters
        ----------
        dataset: Dataset
            The training dataset

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        # 1. Scale the data if required
        if self.scale:
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale X
            X_scaled = (dataset.X - self.mean) / self.std
        else:
            X_scaled = dataset.X

        # 2. Add intercept term to X (column of ones at the beginning)
        m = dataset.X.shape[0]
        X_intercept = np.c_[np.ones(m), X_scaled]

        # 3. Compute the penalty matrix (l2_penalty * identity matrix)
        n = X_intercept.shape[1]
        penalty_matrix = self.l2_penalty * np.eye(n)

        # 4. Change the first position of the penalty matrix to 0
        penalty_matrix[0, 0] = 0

        # 5. Compute the model parameters
        X_T = X_intercept.T

        # A_Matrix = (X^T * X + penalty)
        A_Matrix = X_T.dot(X_intercept) + penalty_matrix

        # Inverse matrix with np.linalg.inv
        inverse_matrix = np.linalg.inv(A_Matrix)

        # theta
        self.theta = inverse_matrix.dot(X_T).dot(dataset.y)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the target values for the given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict

        Returns
        -------
        predictions: np.ndarray
            The predicted target values
        """
        # 1. Scale the data if required
        if self.scale:
            X_scaled = (dataset.X - self.mean) / self.std
        else:
            X_scaled = dataset.X

        # 2. Add intercept term to X
        m = dataset.X.shape[0]
        X_intercept = np.c_[np.ones(m), X_scaled]

        # 3. Compute the predicted Y
        # Since theta includes the intercept at index 0, we can just do the dot product
        predictions = X_intercept.dot(self.theta)

        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the MSE between the predictions and the real values.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate
        predictions: np.ndarray
            The predictions

        Returns
        -------
        mse: float
            The Mean Squared Error
        """
        return mse(dataset.y, predictions)
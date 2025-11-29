from si.base.transformer import Transformer
from si.data.dataset import Dataset
import numpy as np


class PCA(Transformer):
    """
    Principal Component Analysis (PCA) is a linear algebra technique used to reduce the
    dimensions of the dataset.

    It uses eigenvalue decomposition of the covariance matrix of the data.

    parameters:
        - n_components – number of components

    estimated parameters:
        - mean – mean of the samples (features)
        - components – the principal components (a matrix where each row is an
          eigenvector corresponding to a principal component)
        - explained_variance – the amount of variance explained by each principal component
          (a vector of eigenvalues)

    methods:
        - _fit – estimates the mean, principal components, and explained variance
        - _transform – calculates the reduced dataset using the principal components

    """

    def __init__(self, n_components: int, **kwargs):
        """
        Initializes the PCA object, setting the number of components to be retained.

        Parameters
        ----------
        n_components : int
            The number of principal components to keep

        Attributes
        ----------
        mean : np.ndarray or None
            Mean of the samples (features), inferred in _fit
        components : np.ndarray or None
            The principal components matrix (n_components x n_features)
        explained_variance : np.ndarray or None
            The amount of variance explained by each principal component
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.is_fitted = False

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Estimates the mean, principal components, and explained variance using
        eigenvalue decomposition of the covariance matrix

        1. Infer the mean of the samples (features) and subtract it to center the data
        2. Calculate the covariance matrix and perform eigenvalue decomposition
        3. Infer the principal components by selecting the n_components highest eigenvectors
        4. Calculate the explained variance for the selected components

        Parameters
        ----------
        dataset : Dataset
            The dataset used to estimate the principal components.

        Returns
        -------
        self : PCA
            The fitted PCA object.
        """
        # 1. Center the Data
        self.mean = dataset.get_mean()
        X_centered = dataset.X - self.mean

        # 2. Calculate Covariance and Decomposition
        co_matrix = np.cov(X_centered, rowvar=False)  # rowvar=False → columns are the features
        eigen_values, eigen_vectors = np.linalg.eig(co_matrix)  # Returns eigenvalues (lambda) and eigenvectors (V)

        # 3. Sort and Select Indices
        sorted_indices = np.argsort(eigen_values)[::-1][:self.n_components]

        # 4. Infer Principal Components (sorted eigenvectors)
        self.components = eigen_vectors[:, sorted_indices].T  # self.components (n_components x n_features)

        # 5. Infer Explained Variance
        total_variance = np.sum(eigen_values)
        self.explained_variance = eigen_values[sorted_indices] / total_variance

        self.is_fitted = True
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Calculates the reduced dataset by projecting the data onto the principal components.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            A new Dataset object containing the reduced data (X_reduced = X_centered @ V)
        """
        # 1. Center the Data (Using the mean inferred in _fit)
        X_centered = dataset.X - self.mean  # X - mean

        # 2. Calculate Reduced X (Projection)
        # X_reduced = X_centered @ components.T
        X_reduced = np.dot(X_centered, self.components.T)  # Matrix multiplication

        # 3. Create and return a new Dataset
        features = ["PC" + str(i) for i in range(1, self.n_components + 1)]
        return Dataset(X=X_reduced, y=dataset.y, features=features, label=dataset.label)
from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from datasets import DATASETS_PATH

class TestPCA(TestCase):
    """
    Unit tests for the PCA class, verifying its behavior in fitting and transforming data.
    """

    def setUp(self):
        """
        Set up the test environment by loading the dataset and initializing PCA with a specified number of components.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.n_components = 2
        self.pca = PCA(n_components=self.n_components)

    def test_fit(self):
        """
        Tests the PCA fitting process (_fit) to ensure that the mean,
        components and explained variance are computed and stored correctly.
        """

        # The number of original features (Iris) is 4
        n_features_original = self.dataset.X.shape[1]

        # 1. Fit Execution
        self.pca._fit(self.dataset)

        # 2. Verification of self.mean (Mean of Features)
        # self.pca.mean should be the vector of means for each column
        expected_mean = np.mean(self.dataset.X, axis=0)
        self.assertTrue(np.allclose(self.pca.mean, expected_mean))
        self.assertEqual(self.pca.mean.shape, (n_features_original,))

        # 3. Verification of self.components (Principal Components)
        # The shape should be (n_components, n_features_original)
        expected_components_shape = (self.n_components, n_features_original)
        self.assertEqual(self.pca.components.shape, expected_components_shape)

        # 4. Verification of self.explained_variance (Explained Variance)
        # The number of entries should be equal to n_components
        self.assertEqual(len(self.pca.explained_variance), self.n_components)

        # 5. Verification of Explained Variance Limits (Proportion)
        # Individual values should be between 0 and 1, and the sum cannot be greater than 1
        self.assertTrue(np.all(self.pca.explained_variance >= 0) and np.all(self.pca.explained_variance <= 1))

        # 6. Verification of Order (Confirms the Sorting Logic)
        # The first value should be greater than or equal to the second, confirming descending order
        self.assertTrue(self.pca.explained_variance[0] >= self.pca.explained_variance[1])


    def test_transform(self):
        """
        Tests if _transform correctly projects the data to the new dimensional space,
        preserving the number of samples and the type of the returned object.
        """

        # The number of original features (Iris) is 4

        # 1. Fit and Transformation
        self.pca.fit(self.dataset)

        # dataset_transformed is the Dataset object returned by transform()
        dataset_transformed = self.pca.transform(self.dataset)

        # NumPy array with the reduced data
        X_reduced = dataset_transformed.X

        # 2. Critical Assertions

        # A) Type Verification (Should return a Dataset object)
        self.assertIsInstance(dataset_transformed, type(self.dataset))

        # B) Shape Verification (Correct Dimension)
        # Number of rows (samples) should be the same
        self.assertEqual(X_reduced.shape[0], self.dataset.X.shape[0])

        # Number of columns (features) should be equal to n_components (2)
        self.assertEqual(X_reduced.shape[1], self.n_components)

        # C) Mean Verification (Centered Data)
        # The X_reduced result should have mean ~0 in each component (PC)
        # Note: This is a strong verification, but useful for PCA.
        self.assertTrue(np.allclose(np.mean(X_reduced, axis=0), 0.0))

        # D) Feature Names Verification
        expected_features = ['PC1', 'PC2']
        self.assertEqual(dataset_transformed.features, expected_features)
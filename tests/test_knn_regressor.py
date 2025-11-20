from unittest import TestCase
import numpy as np
from datasets import DATASETS_PATH
from si.metrics.rmse import rmse
from si.models.knn_regressor import KNNRegressor
import os
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split


class TestKNNRegressor(TestCase):
    """
    Unit tests for the KNNRegressor class, verifying its ability to fit, predict, and score datasets.
    """

    def setUp(self):
        """
        Prepare the testing environment by loading the dataset.
        """
        # Certifica-te que este caminho está correto no teu sistema
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        """
        Test the fitting process of the KNNRegressor to ensure
        it correctly stores the training data.
        """
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)

        # Verify that the training data is correctly stored in the model
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        """
        Test the prediction functionality of the KNNRegressor
        to ensure it returns predictions of the correct shape.
        """
        # CORREÇÃO AQUI: KNNRegressor em vez de KNNClassifier
        knn = KNNRegressor(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model and make predictions
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        # Validate the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])

        # Sugestão Extra: Verificar se as previsões são números (floats)
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_score(self):
        """
        Test the scoring functionality of the KNNRegressor by comparing
        the calculated RMSE to the expected RMSE.
        """
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)

        # Fit the model, make predictions, and calculate the score
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        score = knn.score(test_dataset)

        # Verify that the computed RMSE matches the expected RMSE
        expected_score = rmse(test_dataset.y, predictions)
        self.assertAlmostEqual(score, expected_score, places=5)
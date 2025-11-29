from unittest import TestCase

import numpy as np
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares


class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)

        # Na solução fechada (Least Squares), o theta inclui o intercepto (bias) na posição 0.
        # Logo, o tamanho do theta deve ser numero_features + 1
        self.assertEqual(ridge.theta.shape[0], self.train_dataset.shape()[1] + 1)

        # Verificamos se a média e desvio padrão foram calculados (se scale=True)
        self.assertNotEqual(len(ridge.mean), 0)
        self.assertNotEqual(len(ridge.std), 0)

        # O theta não deve ser None após o fit
        self.assertIsNotNone(ridge.theta)

    def test_predict(self):
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)

        predictions = ridge.predict(self.test_dataset)

        # O número de previsões deve ser igual ao número de amostras de teste
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

        # As previsões devem ser floats
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_score(self):
        ridge = RidgeRegressionLeastSquares(l2_penalty=1.0, scale=True)
        ridge.fit(self.train_dataset)

        mse_ = ridge.score(self.test_dataset)

        # O MSE deve ser um valor positivo
        self.assertGreaterEqual(mse_, 0.0)

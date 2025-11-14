from unittest import TestCase
from datasets import DATASETS_PATH
import os
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification
import numpy as np
import unittest





class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.X_data = np.array([
            [1, 5, 20, 10],
            [2, 6, 21, 11],
            [3, 7, 22, 12],
            [4, 8, 23, 13]
        ])
        self.y_data = np.array([0, 1, 0, 1])
        self.dataset = Dataset(self.X_data, self.y_data, features=['f1', 'f2', 'f3', 'f4'], label='target')

        self.F_scores = np.array([10.0, 5.0, 20.0, 20.0])
        self.p_values = np.array([0.1, 0.5, 0.01, 0.01])

        def mock_f_classification(dataset):
            return self.F_scores, self.p_values

        self.mock_score_func = mock_f_classification
        self.expected_F = self.F_scores #to test the _fit method
        self.expected_p = self.p_values #to test the _fit method


    def test_fit(self):
        #Test the _fit method to see if it stores the F and p values
        selector = SelectPercentile(percentile=50, score_func=self.mock_score_func)
        selector._fit(self.dataset)
        self.assertTrue(np.allclose(selector.F, self.expected_F))
        self.assertTrue(np.allclose(selector.p, self.expected_p))
        self.assertEqual(selector.F.shape, (4,))


    def test_transform_with_ties_percentile_50(self):
        selector = SelectPercentile(percentile=50, score_func=self.mock_score_func)
        selector.F = self.F_scores
        selector.p = self.p_values

        new_dataset = selector._transform(self.dataset)

        expected_X = np.array([
            [20, 10],
            [21, 11],
            [22, 12],
            [23, 13]
        ])
        expected_features = ['f3', 'f4']

        self.assertEqual(new_dataset.X.shape, (4, 2))
        self.assertTrue(np.allclose(new_dataset.X, expected_X))
        self.assertEqual(new_dataset.features, expected_features)


    def test_transform_without_ties_75(self):
        selector = SelectPercentile(percentile=75, score_func=self.mock_score_func)
        selector.F = self.F_scores #[10.0, 5.0, 20.0, 20.0] -> best 3: f3, f4, f1
        selector.p = self.p_values
        new_dataset = selector._transform(self.dataset)

        expected_X = np.array([
                                [1, 20, 10],
                                [2, 21, 11],
                                [3, 22, 12],
                                [4, 23, 13]
                            ])

        expected_features = ['f1', 'f3', 'f4']

        self.assertEqual(new_dataset.X.shape, (4, 3))
        self.assertTrue(np.allclose(new_dataset.X, expected_X))
        self.assertEqual(set(new_dataset.features), set(expected_features))




class TestSelectPercentile_iris(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector._fit(self.dataset)

        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        self.assertEqual(len(selector.F), len(self.dataset.features))
        self.assertEqual(len(selector.p), len(self.dataset.features))

    def test_transform_50(self):
        '''
        >f_classification(self.dataset)[0]
        >[ 119.26450218,   47.3644614 , 1179.0343277 ,  959.32440573]

        > self.features
        >['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        As 2 features com melhor f_score são "petal_length" e "petal_width"
        com indices 2 e 3 respetivamente
        '''

        expected_X = self.dataset.X[: , 2:] # só as features 2 e 3
        expected_features = ['petal_length', 'petal_width']

        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector._fit(self.dataset)
        dataset = selector._transform(self.dataset)

        self.assertEqual(dataset.X.shape[1],2) #apenas duas colunas = 50%
        self.assertTrue(np.allclose(dataset.X, expected_X))
        self.assertEqual(dataset.features, expected_features)

    def test_transform_75(self):
        '''
        >f_classification(self.dataset)[0]
        >[ 119.26450218,   47.3644614 , 1179.0343277 ,  959.32440573]

        > self.features
        >['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

        As 3 features com melhor f_score são "petal_length", "petal_width" e "sepal_length"
        com indices 2, 3 e 0 respetivamente
        '''
        expected_indices = [0, 2, 3]
        expected_X = self.dataset.X[:, expected_indices]
        expected_features = ['sepal_length', 'petal_length', 'petal_width']

        selector = SelectPercentile(percentile=75, score_func=f_classification)
        selector._fit(self.dataset)
        dataset = selector._transform(self.dataset)

        self.assertEqual(dataset.X.shape[1],3)
        self.assertTrue(np.allclose(dataset.X, expected_X))
        self.assertEqual(dataset.features, expected_features)












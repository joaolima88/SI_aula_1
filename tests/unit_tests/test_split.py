from unittest import TestCase
from datasets import DATASETS_PATH
import os

from si.data.dataset import Dataset
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.model_selection.split import stratified_train_test_split
import numpy as np


class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)




    def test_stratified_train_test_split(self):
        """Testa divisão estratificada básica"""
        train, test = stratified_train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=123
        )

        # Verificar tamanhos
        expected_test_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], expected_test_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - expected_test_size)

        # Verificar proporções
        def get_proportions(y):
            _, counts = np.unique(y, return_counts=True)
            return counts / len(y) * 100

        orig_props = get_proportions(self.dataset.y)
        train_props = get_proportions(train.y)
        test_props = get_proportions(test.y)

        self.assertTrue(np.allclose(orig_props, train_props, rtol=1e-3))
        self.assertTrue(np.allclose(orig_props, test_props, rtol=1e-3))

    def test_stratified_reproducibility(self):
        """Testa reprodutibilidade com mesma seed"""
        train1, test1 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)
        train2, test2 = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

        self.assertTrue(np.array_equal(train1.X, train2.X))
        self.assertTrue(np.array_equal(test1.X, test2.X))


    def test_stratified_invalid_test_size(self):
        """Testa validação de test_size inválido"""
        with self.assertRaises(ValueError):
            stratified_train_test_split(self.dataset, test_size=0)

        with self.assertRaises(ValueError):
            stratified_train_test_split(self.dataset, test_size=1)

        with self.assertRaises(ValueError):
            stratified_train_test_split(self.dataset, test_size=1.5)

    def test_stratified_imbalanced_dataset(self):
        """Testa com dataset desbalanceado"""
        X = np.random.rand(100, 4)
        y = np.array([0] * 90 + [1] * 10)
        dataset_imbalanced = Dataset(X, y)

        train, test = stratified_train_test_split(dataset_imbalanced, test_size=0.2, random_state=42)

        # Verificar proporções
        orig_prop = np.sum(dataset_imbalanced.y == 1) / len(dataset_imbalanced.y)
        train_prop = np.sum(train.y == 1) / len(train.y)
        test_prop = np.sum(test.y == 1) / len(test.y)

        # Tolerância de 5% para classes muito desbalanceadas
        self.assertAlmostEqual(train_prop, orig_prop, delta=0.05)
        self.assertAlmostEqual(test_prop, orig_prop, delta=0.05)



import numpy as np
import unittest
from sklearn.metrics import jaccard_score
from si.statistics.tanimoto_similarity import tanimoto_similarity


class TestTanimotoSimilarity(unittest.TestCase):

    def test_perfect_match(self):
        """Test perfect similarity and total dissimilarity (Tanimoto = [1.0, 0.0])."""
        x = np.array([1, 1, 0, 1])
        y = np.array([[1, 1, 0, 1], [0, 0, 0, 0]])

        # 1. x vs y[0]:
        #    Intersection (common 1s): [P0, P1, P3] -> 3
        #    Union (x_1s + y_1s - common 1s): 3 + 3 - 3 = 3
        #    Tanimoto: 3 / 3 = 1

        # 2. x vs y[1]:
        #    Intersection (common 1s): 0
        #    Union (x_1s + y_1s - common 1s): 0
        #    Tanimoto: 0

        our_similarity = tanimoto_similarity(x, y)
        expected_similarity = np.array([1.0, 0.0])

        self.assertTrue(np.allclose(our_similarity, expected_similarity))

    def test_partial_overlap(self):
        """Test the case of partial overlap (Tanimoto between 0 and 1)."""
        x = np.array([1, 0, 1, 1])
        y = np.array([[1, 1, 0, 1], [0, 0, 0, 1]])

        # 1. x vs y[0]:
        #    Intersection (common 1s): [P0, P3] -> 2
        #    Union (x_1s + y_1s - common 1s): 3 + 3 - 2 = 4
        #    Tanimoto: 2 / 4 = 0.5

        # 2. x vs y[1]:
        #    Intersection (common 1s): [P3] -> 1
        #    Union (x_1s + y_1s - common 1s): 3 + 1 - 1 = 3
        #    Tanimoto: 1 / 3

        our_similarity = tanimoto_similarity(x, y)
        expected_similarity = np.array([0.5, 1 / 3])

        self.assertTrue(np.allclose(our_similarity, expected_similarity))

    def test_division_by_zero_case(self):
        """Test the edge case of division by zero (both vectors are all zeros)."""
        x = np.array([0, 0, 0, 0])
        y = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])

        our_similarity = tanimoto_similarity(x, y)
        expected_similarity = np.array([0.0, 0.0])

        self.assertTrue(np.allclose(our_similarity, expected_similarity))

    def test_against_sklearn_jaccard(self):
        """Verify accuracy against Scikit-learn reference."""
        x_test = np.array([1, 1, 0, 1])
        y_test = np.array([[1, 0, 1, 1], [0, 1, 0, 0]])

        j1 = jaccard_score(x_test, y_test[0])  # 0.5
        j2 = jaccard_score(x_test, y_test[1])  # 1/3

        our_similarity = tanimoto_similarity(x_test, y_test)
        sklearn_similarity = np.array([j1, j2])

        self.assertTrue(np.allclose(our_similarity, sklearn_similarity))
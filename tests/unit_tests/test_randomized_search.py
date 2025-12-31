from unittest import TestCase
import os
import numpy as np
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
from datasets import DATASETS_PATH

class TestRandomizedSearchCV(TestCase):
    """
    Unit tests for the randomized_search_cv function.
    """

    def setUp(self):
        """
        Set up the testing environment by loading the dataset used for validation.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_k_fold_cross_validation(self):
        """
        Test the randomized_search_cv function to validate:
        - The number of scores returned matches the expected number of iterations.
        - The hyperparameter grid generates the correct number of best parameters.
        - The best score matches the expected accuracy value.
        """

        np.random.seed(42)

        # Initialize the logistic regression model
        model = LogisticRegression()

        # Define the hyperparameter grid
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200)
        }

        # Perform randomized search with 3-fold cross-validation
        results = randomized_search_cv(model=model,
                                        dataset=self.dataset,
                                        hyperparameter_grid=parameter_grid,
                                        cv=3,
                                        n_iter=10
                                        )

        # Validate that the number of scores matches the number of iterations
        self.assertEqual(len(results["scores"]), 10,
                         "The number of scores does not match the number of iterations.")

        # Validate that the best hyperparameters dictionary contains three keys
        best_hyperparameters = results['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3,
                         "The best hyperparameters do not contain the expected number of parameters.")

        # Validate that the best score matches the expected rounded value
        best_score = results['best_score']
        self.assertEqual(np.round(best_score, 2), 0.97,
                         "The best score does not match the expected value (0.97).")
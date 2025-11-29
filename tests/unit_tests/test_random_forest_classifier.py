from unittest import TestCase
import os
from si.models.random_forest_classifier import RandomForestClassifier
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH


class TestRandomForest(TestCase):
    """
    Unit tests for the RandomForestClassifier class, ensuring its fit, predict, and score functionalities work as expected.
    """

    def setUp(self):
        """
        Set up the testing environment by loading the dataset and splitting it into training and testing sets.
        """

        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        """
        Test the fit method of RandomForestClassifier to verify that the model parameters are initialized correctly.
        """
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)

        # Validate default hyperparameters after initialization
        self.assertEqual(random_forest.min_sample_split, 5, "Default min_sample_split is incorrect.")
        self.assertEqual(random_forest.max_depth, 10, "Default max_depth is incorrect.")


    def test_predict(self):
        """
        Test the predict method to ensure it returns predictions with the correct shape.
        """
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)

        # Generate predictions for the test dataset
        predictions = random_forest.predict(self.test_dataset)

        # Verify the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0], "The number of predictions does not match the number of test samples.")

    def test_score(self):
        """
        Test the score method to verify that the calculated accuracy matches the expected value.
        """
        random_forest = RandomForestClassifier()
        random_forest.fit(self.train_dataset)

        # Calculate the accuracy of the model
        accuracy_ = random_forest.score(self.test_dataset)

        # Validate the accuracy score
        self.assertGreater(accuracy_, 0.85)
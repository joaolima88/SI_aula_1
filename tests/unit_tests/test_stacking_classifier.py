from unittest import TestCase
import os
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH

from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression


class TestStackingClassifier(TestCase):
    """
    Unit tests for the StackingClassifier class, verifying its fit, predict, and score methods.
    """

    def setUp(self):
        """
        Prepare the dataset and split it into training and testing sets for use in the tests.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):
        """
        Test the fit method to ensure the pipeline runs without errors.
        """
        knn = KNNClassifier()
        logistic = LogisticRegression()
        tree = DecisionTreeClassifier(seed=42)
        final_knn = KNNClassifier()

        stacking = StackingClassifier(models=[knn, logistic, tree], final_model=final_knn)
        stacking.fit(self.train_dataset)

        self.assertEqual(stacking.models[2].min_sample_split, 2,
                         "DecisionTreeClassifier min_sample_split is incorrect.")
        self.assertEqual(stacking.models[2].max_depth, 10, "DecisionTreeClassifier max_depth is incorrect.")
        self.assertTrue(stacking.models[0].dataset is not None)


    def test_predict(self):
        """
        Test the predict method to ensure it returns predictions with the correct shape.
        """
        # Initialize base models and final model
        knn = KNNClassifier()
        logistic = LogisticRegression()
        tree = DecisionTreeClassifier(seed=42)
        final_knn = KNNClassifier()

        # Create and train the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic, tree], final_model=final_knn)
        stacking_classifier.fit(self.train_dataset)

        # Generate predictions for the test dataset
        predictions = stacking_classifier.predict(self.test_dataset)

        # Validate that the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0],
                         "The number of predictions does not match the number of test samples.")

    def test_score(self):
        knn = KNNClassifier()
        logistic = LogisticRegression()
        tree = DecisionTreeClassifier(seed=42)
        final_knn = KNNClassifier()

        stacking = StackingClassifier(models=[knn, logistic, tree], final_model=final_knn)
        stacking.fit(self.train_dataset)

        accuracy_ = round(stacking.score(self.test_dataset), 4)
        self.assertGreater(accuracy_, 0.90)
        print(f"Accuracy Obtida: {accuracy_}")
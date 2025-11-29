import numpy as np
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.base.model import Model

class RandomForestClassifier(Model):
    """
    Random Forest is an ensemble learning method that builds multiple decision trees
    and combines their predictions to enhance accuracy and minimize overfitting.
    """

    def __init__(self, n_estimators: int = 100, max_features: int = None,
                 min_sample_split: int = 5, max_depth: int = 10,
                 mode: str = "gini", seed: int = 123, **kwargs):
        """
        Initialize the Random Forest Classifier with specified hyperparameters.

        Parameters
        ----------
        n_estimators : int
            Number of decision trees in the forest.
        max_features : int
            Maximum number of features to consider for each split.
        min_sample_split : int
            Minimum number of samples required to split a node.
        max_depth : int
            Maximum depth of the trees.
        mode : str
            Impurity calculation mode ("gini" or "entropy").
        seed : int
            Random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode if mode in {"gini", "entropy"} else "gini"
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset) -> "RandomForestClassifier":
        """
        Train the Random Forest by building decision trees on bootstrap samples.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        RandomForestClassifier
            The trained Random Forest model.
        """
        np.random.seed(self.seed)

        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.shape()[1]))

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            sample_indices = np.random.choice(dataset.shape()[0], size=dataset.shape()[0], replace=True)
            feature_indices = np.random.choice(dataset.shape()[1], size=self.max_features, replace=False)
            bootstrap_features = [dataset.features[i] for i in feature_indices]

            # Create bootstrap dataset
            bootstrap_dataset = Dataset(
                X=dataset.X[sample_indices][:, feature_indices],
                y=dataset.y[sample_indices],
                features=bootstrap_features,
                label=dataset.label
            )

            # Train decision tree
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)

            # Store tree and its features
            self.trees.append((bootstrap_features, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:

        """
        Predict target values for a dataset using the Random Forest.

        Parameters
        ----------
        dataset : Dataset
            Dataset for prediction.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        predictions = []
        # Iterate over each trained tree and the specific features it uses
        for features, tree in self.trees:
            # Find the indices of the columns in the test dataset that correspond
            # to the features this specific tree was trained on.
            feature_indices = [dataset.features.index(feature) for feature in features]

            # Select and reorder the columns of the test data
            X_subset = dataset.X[:, feature_indices]

            # Create a temporary dataset
            subset = Dataset(
                X=X_subset,
                y=dataset.y,
                features=features,
                label=dataset.label
            )

            # Get the predictions from the current tree and append them to the list
            predictions.append(tree.predict(subset))

        # Matrix transposition
        predictions = np.array(predictions).T

        def counts(row):
            labels, counts_ = np.unique(row, return_counts=True)
            return labels[np.argmax(counts_)]

        return np.apply_along_axis(counts, axis=1, arr=predictions)


    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate the accuracy of the Random Forest.

        Parameters
        ----------
        dataset : Dataset
            Dataset with true labels.
        predictions : np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return accuracy(dataset.y, predictions)
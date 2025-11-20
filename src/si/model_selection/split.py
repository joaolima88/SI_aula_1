from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[
    Dataset, Dataset]:
    """
    Split the dataset into training and test sets while maintaining class proportions.

    This function performs stratified sampling to ensure that both training and test sets
    have approximately the same class distribution as the original dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be split into training and test sets
    test_size : float, optional
        Proportion of the dataset to include in the test split (default: 0.2)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Tuple[Dataset, Dataset]
        Tuple containing the training dataset and test dataset

    Raises
    ------
    ValueError
        If test_size is not between 0 and 1, or if dataset doesn't have labels
    """
    # Input validation
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not dataset.has_label():
        raise ValueError("Dataset must have labels for stratified split")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Identify unique classes in the dataset
    unique_labels = np.unique(dataset.y)

    # Initialize lists to store indices for training and test sets
    train_indices = []
    test_indices = []

    # Perform stratified sampling for each class
    for label in unique_labels:
        # Get indices for current class
        label_mask = dataset.y == label
        label_indices = np.where(label_mask)[0]

        # Shuffle indices for current class
        np.random.shuffle(label_indices)

        # Calculate number of samples for test set for this class
        n_test = int(len(label_indices) * test_size)

        # Split indices for current class
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])

    # Convert lists to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Shuffle the final sets to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Create new Dataset objects for training and test sets
    train = Dataset(
        X=dataset.X[train_indices],
        y=dataset.y[train_indices],
        features=dataset.features,
        label=dataset.label
    )

    test = Dataset(
        X=dataset.X[test_indices],
        y=dataset.y[test_indices],
        features=dataset.features,
        label=dataset.label
    )

    return train, test
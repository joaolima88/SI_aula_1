import numpy as np
from si.data.dataset import Dataset

def train_train_spilt(dataset:Dataset, test_size:float, random_state:int = 42):

    np.random.seed(random_state)
    n_samples = dataset.shape()[0]
    test_samples = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_indices = permutations[:test_samples]
    train_indices = permutations[test_samples:]

    train_dataset = Dataset(X=dataset.x[train_indices], y=dataset.y[train_indices],
                            features=dataset.features, label = dataset.label)

    test_dataset = Dataset(X=dataset.X[test_indices], y=dataset.y[test_indices],
                           features = dataset.features, label = dataset.label)

    return train_dataset, test_dataset



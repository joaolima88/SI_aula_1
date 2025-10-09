import numpy as np
import scipy
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    dataset – the Dataset object
    Returns
    -------
    output:
    tuple - with F values + tuple with p values


    """

    classes = dataset.get_classes()
    samples_per_class = []

    for class_ in classes:
        mask = dataset.y == class_
        class_X = dataset.X[mask, :]
        samples_per_class.append(class_X)

    F, p = scipy.stats.f_oneway(samples_per_class)
    return F, p
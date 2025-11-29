import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Tanimoto similarity (Jaccard Coefficient)
    of a single binary point (x) to a set of binary points (y)
    using the formula S_J(x, y) = (∑(x_i · y_i)) / (∑x_i + ∑y_i - ∑(x_i · y_i)).

    Parameters
    ----------
    x: np.ndarray
        Point.
    y: np.ndarray
        Set of points.

    Returns
    -------
    np.ndarray
        An array containing the Tanimoto distances between x and each point in y.


    """
    numerator = np.sum(x * y, axis=1) # ∑(x_i · y_i)
    denominator = np.sum(x) + np.sum(y, axis=1) - numerator # ∑x_i + ∑y_i - ∑(x_i · y_i)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


if __name__ == '__main__':
    from sklearn.metrics import jaccard_score
    x_bin = np.array([1, 1, 0])
    y_bin = np.array([[1, 0, 1], [0, 1, 1]])

    our_similarity = tanimoto_similarity(x_bin, y_bin)

    j1 = jaccard_score(x_bin, y_bin[0])
    j2 = jaccard_score(x_bin, y_bin[1])
    sklearn_similarity = np.array([j1, j2])

    assert np.allclose(our_similarity, sklearn_similarity)
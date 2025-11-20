import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between actual and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        Array with the actual target values.
    y_pred : np.ndarray
        Array with the predicted target values.

    Returns
    -------
    float
        The computed RMSE value, indicating the average prediction error.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
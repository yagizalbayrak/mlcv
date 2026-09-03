import numpy as np


def rmse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function should calculate the root mean squared error given target y and prediction y_pred

    Args:
        - y(np.array): target data
        - y_pred(np.array): predicted data

    Returns:
        - err (float): root mean squared error between y and y_pred

    """

    # RMSE = sqrt( (1/N) * sum( (y_i - y_pred_i)^2 ) )
    err = np.sqrt(np.mean((y - y_pred) ** 2))

    return err


def split_data(
    x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training and test sets (randomly)

    Args:
        - x: input data
        - y: target data
        - train_ratio: the ratio of the train set (about train_ratio of the data will be used for training)

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validating
        - y_val: target data used for validating

    """
    N = len(x)

    # Use a fixed random seed for reproducibility
    np.random.seed(42)

    # Generate a random permutation of indices
    indices = np.random.permutation(N)

    # Calculate the split point
    split_idx = int(N * train_ratio)

    # Split indices into training and validation
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_val = x[val_indices]
    y_val = y[val_indices]

    return x_train, y_train, x_val, y_val

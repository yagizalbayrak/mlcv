import numpy as np


def split_data(x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    This function should split the X and Y data in training and test sets
    This function has already been implemented in Exercise 2 so you do not need to edit it here

    Args:
        - x: input data
        - y: target data

    Returns:
        - x_train: input data used for training
        - y_train: target data used for training
        - x_val: input data used for validation
        - y_val: target data used for validation

    """
    # split the train data
    train_index = np.random.choice(len(x), int(len(x) * 0.5), replace=False)
    train_set = np.zeros((x.shape[0])).astype(bool)
    train_set[train_index] = True

    x_train = x[train_set]
    x_val = x[~train_set]
    y_train = y[train_set]
    y_val = y[~train_set]

    return x_train, y_train, x_val, y_val


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    This function should calculate the sigmoid activation function w.r.t. x

    Args:
        - x (np.ndarray): vector with float values to calculate the sigmoid function

    Returns:
        - sigmoid_x (np.ndarray): output vector with the values sigmoid(x)
    """
    # sigmoid(x) = 1 / (1 + e^(-x))
    sigmoid_x = 1.0 / (1.0 + np.exp(-x))

    return sigmoid_x


def softmax(x: np.ndarray) -> np.ndarray:
    """
    This function should calculate the softmax activation function w.r.t. x

    Args:
        - x (np.ndarray): vector with float values to calculate the softmax function

    Returns:
        - softmax_x (np.ndarray): output vector with the values softmax(x)
    """
    # softmax(x_i) = e^(x_i) / sum(e^(x_j))
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    softmax_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    return softmax_x

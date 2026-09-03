import numpy as np
from mlrcv.core import *
from mlrcv.utils import *
import typing


class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        y_pred = sigmoid(x.dot(self.theta))

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der (np.ndarray): first derivative value
        """
        der = x.T.dot(y_pred - y)

        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta parameters that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """
        n = x.shape[0]
        x = np.hstack([np.ones((n, 1)), x])

        # Theta randomly initialize
        self.theta = np.random.rand(x.shape[1], 1)

        # Gradient descent loop
        for epoch in range(self.epochs):
            # 1. Forward pass: predict y
            y_pred = self.predict_y(x)

            # 2. Gradient calculation
            gradient = self.first_derivative(x, y_pred, y)

            # 3. Update theta parameters
            self.theta = self.theta - self.learning_rate * gradient

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred))
            note: accuracy function already implemented in core.py
        """
        # Add bias term to x
        n = x.shape[0]
        x = np.hstack([np.ones((n, 1)), x])

        # Predict y using the model
        y_pred = self.predict_y(x)
        y_pred_class = (y_pred >= 0.5).astype(int)

        # Calculate accuracy
        acc = accuracy(y, y_pred_class)

        return acc


class MultiClassLogisticRegression:
    def __init__(self, learning_rate: float, epochs: int):
        """
        This function should initialize the model parameters

        Args:
            - learning_rate (float): the lambda value to multiply the gradients during the training parameters update
            - epochs (int): number of epochs to train the model

        Returns:
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_class = None

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y class given an input x

        Args:
            - x (np.ndarray): input data to predict y classes

        Returns:
            - y_pred (np.ndarray): the model prediction of the input x
        """
        num_classes = len(self.theta_class)
        z = np.hstack([x.dot(self.theta_class[k]) for k in range(num_classes)])
        y_pred = softmax(z)

        return y_pred

    def first_derivative(self, x: np.ndarray, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        This function should calculate the first derivative w.r.t. input x, predicted y and true labels y,
        for each possible class.

        Args:
            - x (np.ndarray): input data
            - y_pred (np.ndarray): predictions of x
            - y (np.ndarray): true labels of x

        Returns:
            - der: first derivative value
        """
        der = x.T.dot(y_pred - y)

        return der

    def train_model(self, x: np.ndarray, y: np.ndarray):
        """
        This function should use train the model to find theta_class parameters (multiclass) that best fit the data

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x
        """
        # Add bias term to x
        n = x.shape[0]
        num_classes = y.shape[1]  # one-hot encoded
        x = np.hstack([np.ones((n, 1)), x])  # (n, 3)

        num_features = x.shape[1]  # 3 (bias + x1 + x2)

        # Theta randomly initialize for each class
        self.theta_class = [np.random.rand(num_features, 1) for _ in range(num_classes)]

        # Gradient descent loop
        for epoch in range(self.epochs):
            # 1. Forward pass: predict y
            y_pred = self.predict_y(x)

            # 2. Gradient calculation for each class
            gradient = self.first_derivative(x, y_pred, y)  # (num_features, num_classes)

            # 3. Update theta for each class
            for k in range(num_classes):
                self.theta_class[k] = self.theta_class[k] - self.learning_rate * gradient[:, k:k+1]

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        This function should use evaluate the model and output the accuracy of the model
        (accuracy function already implemented)

        Args:
            - x (np.ndarray): input data to predict y classes
            - y (np.ndarray): true labels of x

        Returns:
            - acc (float): accuracy of the model (accuracy(y,y_pred))
            note: accuracy function already implemented in core.py
        """
        # Add bias term to x
        n = x.shape[0]
        x = np.hstack([np.ones((n, 1)), x])

        # 
        y_pred = self.predict_y(x)
        acc = accuracy(y, y_pred)

        return acc

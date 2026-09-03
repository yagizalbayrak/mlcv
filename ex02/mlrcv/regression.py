import numpy as np
from typing import Optional


class LinearRegression:
    def __init__(self):
        self.theta_0 = None
        self.theta_1 = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray):
        """
        This function should calculate the parameters theta0 and theta1 for the regression line

        Args:
            - x (np.array): input data
            - y (np.array): target data

        """
        N = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # Calculate theta_1 (slope)
        self.theta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

        # Calculate theta_0 (intercept)
        self.theta_0 = y_mean - self.theta_1 * x_mean

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta0 and theta1 to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y_pred: y computed w.r.t. to input x and model theta0 and theta1

        """

        # f(x) = x * theta_1 + theta_0
        y_pred = x * self.theta_1 + self.theta_0

        return y_pred


class NonLinearRegression:
    def __init__(self):
        self.theta = None

    def calculate_theta(self, x: np.ndarray, y: np.ndarray, degree: Optional[int] = 2):
        """
        This function should calculate the parameters theta for the regression curve.
        In this case there should be a vector with the theta parameters (len(parameters)=degree + 1).

        Args:
            - x: input data
            - y: target data
            - degree (int): degree of the polynomial curve

        Returns:

        """
    
        # Build the design matrix Phi (Vandermonde matrix) for polynomial regression:
        # Phi = [1, x, x^2, x^3, ..., x^degree]
        # Each row i: [1, x_i, x_i^2, ..., x_i^degree]
        # This is an N x (degree+1) matrix.
        N = len(x)
        Phi = np.zeros((N, degree + 1))
        for d in range(degree + 1):
            Phi[:, d] = x ** d

        # Closed-form solution (normal equation):
        # theta = (Phi^T Phi)^(-1) Phi^T y
        self.theta = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        This function should use the parameters theta to predict the y value given an input x

        Args:
            - x: input data

        Returns:
            - y: y computed w.r.t. to input x and model theta parameters
        """

        # Infer degree from the length of theta (len = degree + 1)
        degree = len(self.theta) - 1

        # Build the design matrix Phi for the given x
        N = len(x)
        Phi = np.zeros((N, degree + 1))
        for d in range(degree + 1):
            Phi[:, d] = x ** d

        # Prediction: y = Phi @ theta
        y_pred = Phi @ self.theta

        return y_pred

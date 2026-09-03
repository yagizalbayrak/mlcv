import numpy as np
import pandas as pd
from mlrcv.core import *
from typing import Optional


class KMeans:
    def __init__(self, k_clusters: int, max_iter: Optional[int] = 300, init_method: Optional[str] = "rand"):
        """
        This function initializes the KMeans method:

        Args:
            - k_clusters (int): number of cluster to divide your data
            - max_iter (int): max number of iterations to define the clusters
            - init_method (str): method to initialize the cluster centers (rand or k++)

        Returns:
        """
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.k_centers = None
        self.init_method = init_method
        self.init_dict = {
            "rand": self.init_random_centers,
            "k++": self.init_centers_kpp,
            "fixed": self.init_fixed_centers,
        }

    def euclidean_distance(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        This function computes the euclidean distance between every point in x to the center:

        Args:
            - x (np.ndarray): input points x, shape (N, D), in our case, D=2
            - center (np.ndarray): cluster center to compute the euclidean distance, shape (D,)

        Returns:
           - dist (np.ndarray): the euclidean distance of each x point to the center, shape (N,)
        """

        # dist: (N,)
        dist = np.sqrt(np.sum((x - center) ** 2, axis=1))

        return dist

    def init_fixed_centers(self, _):
        """
        No need to implement anything, this just initialize the centers with an example of failing case when initializing with random seeds
        """
        self.k_centers = np.array([[0.12334165, -0.04550881], [0.13236559, -1.42220759], [0.12894003, -1.42228261]])

    def init_centers_kpp(self, x: np.ndarray):
        """
        This function initialize the clusters centers using the KMeans++ method (farthest point sampling). Instead of
        randomly initialize the centers this method gets the first center as a random sample of x then computes the
        of every other x sample to the already defined centers, then use the calculated distance as a probability
        distribution to pick other sample as the next center (repeat until define the K initial centers):

        Args:
            - x (np.ndarray): input points x, shape (N, D)

        Returns:
            Sets self.k_centers: (K, D)
        """

        # implement here your function
        # Hints:
        # Step 1: Pick one random data point as the first center (np.random.choice)
        # Step 2: Loop until you have self.k_clusters centers:
        #   a) Compute distance from every point to each existing center using self.euclidean_distance()
        #   b) For each point, keep only the min distance to any existing center (np.min)
        #   c) Normalize distances to get a probability distribution (farther points → higher probability)
        #   d) Use cumulative sum (np.cumsum) for weighted random sampling of the next center
        # Step 3: Convert self.k_centers from list to np.ndarray of shape (K, D)

        N = x.shape[0]
        first_idx = np.random.choice(N)
        centers = [x[first_idx]]

        for _ in range(1, self.k_clusters):
            dists = np.array([self.euclidean_distance(x, c) for c in centers])
            min_dists = np.min(dists, axis=0)
            probs = min_dists / min_dists.sum()
            cumulative = np.cumsum(probs)
            r = np.random.rand()
            idx = np.searchsorted(cumulative, r)
            centers.append(x[idx])

        self.k_centers = np.array(centers)

        """
        plot the initial centers, should be at the end of your function so you can see where are the initial centers
        """
        print("Initial centers from k++ initialization")
        plot_clusters(x, self)

    def init_random_centers(self, x: np.ndarray):
        """
        This function randomly initializes the K initial centers

        Args:
            - x (np.ndarray): input points x, shape (N, D)

        Returns:
            Sets self.k_centers: (K, D)
        """

        # implement here your function
        # Hints:
        # For each feature dimension d in range(D):
        #   Get the min and max of x[:, d]
        #   Sample self.k_clusters random values in that range (np.random.uniform)
        # Stack results into self.k_centers of shape (K, D)

        D = x.shape[1]
        centers = np.zeros((self.k_clusters, D))
        for d in range(D):
            min_val = x[:, d].min()
            max_val = x[:, d].max()
            centers[:, d] = np.random.uniform(min_val, max_val, self.k_clusters)
        self.k_centers = centers

        """
        plot the initial centers, should be at the end of your function so you can see where are the initial centers
        """
        print("Initial centers from random initialization")
        plot_clusters(x, self)

    def fit(self, x: np.ndarray):
        """
        This function uses the input x data to define the K clusters centers. Iterate over x, compute the points
        distances to the center, assign the clusters points, and update the clusters centers (repeat until convergence
        or reach the max_iter):

        Args:
            - x (np.ndarray): input points x, shape (N, D)

        Returns:
            Updates self.k_centers: (K, D)
        """
        
        # implement here your function
        # Hints:
        # N: number of points, D: dimension of points, K: number of clusters
        # Step 1: Initialize centers using self.init_dict[self.init_method](x)
        # Step 2: Loop until convergence or max_iter:
        #   a) Compute distances from each point to each center using self.euclidean_distance()
        #   b) Assign each point to the nearest center using np.argmin()
        #   c) Update each center to the mean of its assigned points using np.mean(..., axis=0)
        #      Handle empty clusters by reinitializing with a random point
        #   d) Check convergence: stop if centers do not change (np.all(new == old))

        """
        you can use the plot_clusters(x, self) function call inside your training loop to check how the
        clusters behave while updating
        """
        # plot_clusters(x, self)
        self.init_dict[self.init_method](x)
        N = x.shape[0]

        for _ in range(self.max_iter):
            dists = np.array([self.euclidean_distance(x, c) for c in self.k_centers])
            assignments = np.argmin(dists, axis=0)

            new_centers = np.copy(self.k_centers)
            for k in range(self.k_clusters):
                points = x[assignments == k]
                if len(points) > 0:
                    new_centers[k] = np.mean(points, axis=0)
                else:
                    new_centers[k] = x[np.random.choice(N)]

            if np.allclose(new_centers, self.k_centers):
                break
            self.k_centers = new_centers

    def predict_c(self, x: np.ndarray) -> np.ndarray:
        """
        This function predicts to which cluster each point in x belongs

        Args:
            - x (np.ndarray): input points x, shape (N, D)

        Returns:
            - c_pred (np.ndarray): assigned clusters to each point in x, shape (N,) integers in [0, K)
        """
        
        # c_pred: (N,)
        dists = np.array([self.euclidean_distance(x, c) for c in self.k_centers])
        c_pred = np.argmin(dists, axis=0)

        return c_pred

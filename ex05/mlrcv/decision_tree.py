import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlrcv.core import *


class TreeNode:
    def __init__(self, node_id: str, max_degree: int):
        """
        Initializes the TreeNode (structure is set up for you; implement the methods below).

        Args:
            - node_id (str): node identifier for this node
            - max_degree (int): recursion depth ceiling (counter starts at 0 at root; not the branching factor).
              When split_node's depth reaches max_degree it stops splitting and becomes a leaf.

        Returns:
        """
        self.max_attr_gain = -np.inf
        self.attr_split = None
        self.attr_split_val = None
        self.node_class = None
        self.children = {}
        self.leaf = False
        self.node_id = node_id
        self.max_degree = max_degree

    def infer_node(self, x: np.ndarray) -> float:
        """
        Traverse the tree with one sample ``x`` and return the leaf class stored at the terminal node.

        Args:
            - x (np.ndarray): one sample, shape (D,)

        Returns:
            - node_class (float): predicted class label for this leaf path (scalar)

        Hint: Walk branches until you hit a leaf; return its stored class (splitting directions must agree with training).
        """
        # node_class: scalar matching y dtype
        if self.leaf:
            return self.node_class

        if x[self.attr_split] <= self.attr_split_val:
            return self.children["left"].infer_node(x)
        else:
            return self.children["right"].infer_node(x)

    def split_node(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        Uses the subset ``(x, y)`` at this node to choose a left/right partition by information gain, then grows children.

        Splitting stops once ``degree`` reaches ``max_degree``, or trivial cases force a leaf.

        Args:
            - x (np.ndarray): subset of inputs at this node, shape (N, D)
            - y (np.ndarray): labels for those rows, shape (N,)
            - degree (int): current depth (0 at root)

        Returns:
            None (updates this node and subtree in place).

        Hint: Decide when this node stays a leaf (labels / depth limits); otherwise pick best feature split via attr_gain,
        partition the data consistently, recurse on children.
        """
        unique_classes = np.unique(y)
        # majority class
        counts = np.bincount(y.astype(int))
        self.node_class = np.argmax(counts)

        if degree >= self.max_degree or len(unique_classes) == 1:
            self.leaf = True
            return

        best_gain = -np.inf
        best_attr = None
        best_val = None

        for col in range(x.shape[1]):
            gain, val = self.attr_gain(x[:, col], y)
            if gain is not None and gain > best_gain:
                best_gain = gain
                best_attr = col
                best_val = val

        if best_gain <= 0 or best_attr is None:
            self.leaf = True
            return

        self.attr_split = best_attr
        self.attr_split_val = best_val
        self.max_attr_gain = best_gain

        left_mask = x[:, best_attr] <= best_val
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            self.leaf = True
            return

        left_child = TreeNode(self.node_id + "_L", self.max_degree)
        right_child = TreeNode(self.node_id + "_R", self.max_degree)

        self.children["left"] = left_child
        self.children["right"] = right_child

        left_child.split_node(x[left_mask], y[left_mask], degree + 1)
        right_child.split_node(x[right_mask], y[right_mask], degree + 1)

    def attr_gain(self, x_attr: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Candidate thresholds on one feature column; returns the best split using information gain:

        Args:
            - x_attr (np.ndarray): one feature column, shape (N,)
            - y (np.ndarray): labels for those rows, shape (N,)

        Returns:
            - split_gain (float): best information gain among candidate thresholds (scalar)
            - split_value (float): chosen threshold on x_attr (scalar)

        Hint: Try several thresholds spaced along `[min(x_attr), max(x_attr)]` (for example, 10 steps);
        evaluate information gain each time and keep the best gain and its threshold (skip empty sides).
        """
        split_gain = -np.inf
        split_value = None

        min_val = np.min(x_attr)
        max_val = np.max(x_attr)

        if min_val == max_val:
            return None, None

        thresholds = np.linspace(min_val, max_val, 52)[1:-1]  # 50 interior thresholds

        for t in thresholds:
            left_mask = x_attr <= t
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            gain = self.information_gain(y, y[left_mask], y[right_mask])
            if gain > split_gain:
                split_gain = gain
                split_value = t

        return split_gain, split_value

    def information_gain(self, y: np.ndarray, y_l: np.ndarray, y_r: np.ndarray) -> float:
        """
        Information gain for a candidate split comparing parent labels to left/right subsets:

        Args:
            - y (np.ndarray): all labels at this node, shape (N,)
            - y_l (np.ndarray): labels assigned to left child bucket, shape (N_l,)
            - y_r (np.ndarray): labels assigned to right child bucket, shape (N_r,)

        Returns:
            - I (float): information gain (scalar)

        Hint: Compare parent entropy to the weighted entropy of left/right subsets (gain = reduction).
        """
        N = len(y)
        N_l = len(y_l)
        N_r = len(y_r)

        # I: scalar
        I = self.entropy(y) - (N_l / N) * self.entropy(y_l) - (N_r / N) * self.entropy(y_r)

        return I

    def entropy(self, y: np.ndarray) -> float:
        """
        Shannon entropy of the discrete label distribution of ``y``.

        Args:
            - y (np.ndarray): label set, shape (N,)

        Returns:
            - H (float): entropy (scalar)

        Hint: Discrete label distribution → probabilities → Shannon entropy −Σ p log p.
        """
        # H: scalar
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        H = -np.sum(probs * np.log2(probs + 1e-12))

        return H


class DecisionTree:
    def __init__(self, num_class: int, max_degree: int):
        """
        Initializes the DecisionTree (fields below are set up for you).

        Args:
            - num_class (int): expected number of label classes (kept for API symmetry; splits still come from observed ``y``)
            - max_degree (int): recursion depth ceiling (passed to TreeNode.split_node via root; root starts at depth 0).

        Returns:
        """
        self.root = None
        self.max_degree = max_degree
        self.num_class = num_class

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the tree on training data: build a root node and grow the tree by recursive splitting.

        Args:
            - x (np.ndarray): training inputs, shape (N, D)
            - y (np.ndarray): training labels, shape (N,)

        Returns:
            None (sets self.root).

        Hint: Instantiate the root TreeNode then start recursive splitting from depth 0.
        """
        self.root = TreeNode("root", self.max_degree)
        self.root.split_node(x, y, 0)

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        Predict a class label for every row using the fitted ``TreeNode``.

        Args:
            - x (np.ndarray): inputs to classify, shape (N, D)

        Returns:
            - y_pred (np.ndarray): predicted class per row, shape (N,)

        Hint: Delegate each sample to infer_node walking from the fitted root.
        """
        # y_pred: (N,)
        y_pred = np.array([self.root.infer_node(x[i]) for i in range(x.shape[0])])

        return y_pred

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluates the model: compares ``y`` to ``predict_y(x)`` via ``accuracy`` from ``core``.

        Args:
            - x (np.ndarray): inputs, shape (N, D)
            - y (np.ndarray): true labels, shape (N,)

        Returns:
            - acc (float): accuracy in [0, 100]

        Note: Implemented for you once ``predict_y`` works correctly.
        """
        return accuracy(y, self.predict_y(x))

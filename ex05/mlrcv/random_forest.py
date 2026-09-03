import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlrcv.core import *
from mlrcv.decision_tree import *
from typing import Optional


class RandomTreeNode:
    def __init__(self, node_id: str, max_degree: int):
        """
        Initializes a random-tree node (fields set up like ``TreeNode``).

        Args:
            - node_id (str): node identifier for this node
            - max_degree (int): recursion depth ceiling (starts at root with degree 0; limits how deep splits run).

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
        Traverse from this node until a leaf; return the predicted class stored there.

        Args:
            - x (np.ndarray): one sample, shape (D,)

        Returns:
            - node_class (float): predicted class label for this leaf path (scalar)

        Hint: Same traversal idea as DecisionTree (leaf vs internal split routing).
        """
        if self.leaf:
            return self.node_class

        if x[self.attr_split] <= self.attr_split_val:
            return self.children["left"].infer_node(x)
        else:
            return self.children["right"].infer_node(x)

    def split_node(self, x: np.ndarray, y: np.ndarray, degree: int):
        """
        Same recursive splitting idea as ``TreeNode.split_node``, but thresholds are randomized (see ``attr_gain``).

        Args:
            - x (np.ndarray): subset of inputs at this node, shape (N, D)
            - y (np.ndarray): labels for those rows, shape (N,)
            - degree (int): current depth (0 at root)

        Returns:
            None (updates this node and subtree in place).

        Hint: Analogous recursion to DecisionTree but using RandomTreeNode / its attr_gain.
        """
        unique_classes = np.unique(y)
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

        left_child = RandomTreeNode(self.node_id + "_L", self.max_degree)
        right_child = RandomTreeNode(self.node_id + "_R", self.max_degree)

        self.children["left"] = left_child
        self.children["right"] = right_child

        left_child.split_node(x[left_mask], y[left_mask], degree + 1)
        right_child.split_node(x[right_mask], y[right_mask], degree + 1)

    def attr_gain(self, x_attr: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Builds candidate thresholds like the decision tree, then evaluates information gain only on a **random subset**:

        Args:
            - x_attr (np.ndarray): one feature column, shape (N,)
            - y (np.ndarray): labels for those rows, shape (N,)

        Returns:
            - split_gain (float): best information gain among subsampled candidate thresholds (scalar)
            - split_value (float): chosen threshold on x_attr (scalar)

        Hint: Build the same candidate thresholds as the decision tree, then randomly keep only a small subset of them
              before picking the best information gain.
        """
        split_gain = -np.inf
        split_value = None

        min_val = np.min(x_attr)
        max_val = np.max(x_attr)

        if min_val == max_val:
            return None, None

        thresholds = np.linspace(min_val, max_val, 52)[1:-1]  # 50 interior thresholds

        # randomly select a subset of thresholds
        subset_size = max(1, len(thresholds) // 2)
        rand_indices = np.random.choice(len(thresholds), subset_size, replace=False)
        thresholds = thresholds[rand_indices]

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
        Information gain for a candidate split (parent vs left/right label subsets):

        Args:
            - y (np.ndarray): all labels at this node, shape (N,)
            - y_l (np.ndarray): labels assigned to left child bucket, shape (N_l,)
            - y_r (np.ndarray): labels assigned to right child bucket, shape (N_r,)

        Returns:
            - I (float): information gain (scalar)

        Hint: Same information gain as in ``TreeNode.information_gain`` (reuse the formula).
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

        Hint: Same discrete entropy computation as DecisionTree.
        """
        # H: scalar
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        H = -np.sum(probs * np.log2(probs + 1e-12))

        return H


class RandomTree:
    def __init__(self, num_class: int, max_degree: int):
        """
        Initializes RandomTree (root and counters set up).

        Args:
            - num_class (int): stored for parity with ``DecisionTree`` (single-tree inference still derives classes from labels in ``y``)
            - max_degree (int): recursion depth ceiling for splits (parallel to DecisionTree.max_degree).

        Returns:
        """
        self.root = None
        self.max_degree = max_degree
        self.num_class = num_class

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fits one random tree: create a root ``RandomTreeNode`` and grow it by recursive splitting.

        Args:
            - x (np.ndarray): training inputs, shape (N, D)
            - y (np.ndarray): training labels, shape (N,)

        Returns:
            None (sets self.root).

        Hint: Root RandomTreeNode, then recurse like DecisionTree.fit.
        """
        self.root = RandomTreeNode("root", self.max_degree)
        self.root.split_node(x, y, 0)

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        Predict one class per row using the fitted ``RandomTreeNode``.

        Args:
            - x (np.ndarray): inputs to classify, shape (N, D)

        Returns:
            - y_pred (np.ndarray): predicted class per row, shape (N,)

        Hint: Run infer_node row-wise from the fitted root.
        """
        # y_pred: (N,)
        y_pred = np.array([self.root.infer_node(x[i]) for i in range(x.shape[0])])

        return y_pred

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Uses ``accuracy`` from ``core`` on discrete class predictions from ``predict_y``.

        Args:
            - x (np.ndarray): inputs, shape (N, D)
            - y (np.ndarray): true labels, shape (N,)

        Returns:
            - acc (float): accuracy in [0, 100]

        Note: Implemented for you; complete ``predict_y`` first.
        """
        return accuracy(y, self.predict_y(x))


class RandomForest:
    def __init__(
        self, num_class: int, max_degree: int, trees_num: Optional[int] = 10, random_rho: Optional[float] = 1.0
    ):
        """
        Holds an ensemble of ``RandomTree`` models and metadata (fields wired for you).

        Args:
            - num_class (int): ``C`` in ensemble ``predict_y`` output ``(N, C)`` vote / probability rows
            - max_degree (int): recursion depth ceiling for **each** tree (not the ensemble size ``trees_num``).
            - trees_num (int): how many trees vote during prediction.
            - random_rho (float): multiplier for draw count: sample ``int(N * rho)`` training row indices **with replacement**,
              then build ``(x_s, y_s)`` for each tree.

        Returns:
        """
        self.max_degree = max_degree
        self.trees_num = trees_num
        self.random_rho = random_rho
        self.d_trees = [RandomTree(num_class, self.max_degree) for _ in range(trees_num)]
        self.num_class = num_class

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train every tree on a stochastic resample driven by ``random_tree_data``.

        Args:
            - x (np.ndarray): training inputs, shape (N, D)
            - y (np.ndarray): training labels, shape (N,)

        Returns:
            None (fits each tree in self.d_trees).

        Hint: Bootstrap-style resample proportional to rho for each fitted tree then call its fit.
        """
        for tree in self.d_trees:
            x_s, y_s = self.random_tree_data(x, y)
            tree.fit(x_s, y_s)

    def predict_y(self, x: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction: tally each tree's class votes and return a normalized distribution per sample.

        Args:
            - x (np.ndarray): inputs to classify, shape (N, D)

        Returns:
            - y_pred (np.ndarray): class vote distribution per row, shape (N, C) with C = num_class
               (each row sums to ~1; eval uses core.accuracy with argmax over classes).

        Hint: Collect each tree vote, turn counts into nonnegative distribution rows that normalize to probabilities.
        """
        # y_pred: (N, C)
        N = x.shape[0]
        y_pred = np.zeros((N, self.num_class))

        for tree in self.d_trees:
            preds = tree.predict_y(x)
            for i in range(N):
                y_pred[i, int(preds[i])] += 1

        # normalize to probabilities
        row_sums = y_pred.sum(axis=1, keepdims=True)
        y_pred = y_pred / row_sums

        return y_pred

    def random_tree_data(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Produce a stochastic training subset aligned with ``self.random_rho`` (typically ``int(N * rho)``
        sampled indices **with replacement** into ``x`` / ``y``).

        Args:
            - x (np.ndarray): full training batch, shape (N, D)
            - y (np.ndarray): full training labels, shape (N,)

        Returns:
            - x_s (np.ndarray): sampled rows (`N_s` equals number of distinct row indices retained by your masking scheme)
            - y_s (np.ndarray): sampled labels aligned with ``x_s``

        Hint: Randomly redraw training indices **with replacement** (bootstrap-style), keep matching rows and labels.
        """
        N = x.shape[0]
        sample_size = int(N * self.random_rho)
        indices = np.random.choice(N, sample_size, replace=True)

        x_s = x[indices]
        y_s = y[indices]

        return x_s, y_s

    def eval(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Uses ``accuracy`` from ``core``; probabilistic outputs from ``predict_y`` are turned into winners via ``argmax`` per row.

        Args:
            - x (np.ndarray): inputs, shape (N, D)
            - y (np.ndarray): true labels, shape (N,)

        Returns:
            - acc (float): accuracy in [0, 100]

        Note: Implemented for you; finish ``predict_y`` so rows look like probabilities.
        """
        return accuracy(y, self.predict_y(x))

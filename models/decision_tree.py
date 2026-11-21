from dataclasses import dataclass
import numpy as np


@dataclass
class TreeNode:
    prediction: int
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def entropy(counts):
    """Shannon entropy computed from class counts."""
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(totals, 1)

    return -(p * np.log2(np.maximum(p, 1e-12))).sum(axis=-1)


class DecisionTree:
    def __init__(
        self,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features=None,
        n_thresholds=12,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        """Train the decision tree."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError("Expected non-empty X (2D) and y (1D) of equal length")

        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be nonnegative or None")

        if (
            self.n_thresholds < 1
            or self.min_samples_leaf < 1
            or self.min_samples_split < 2
        ):
            raise ValueError("Invalid split parameters")

        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")

        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.rng_ = np.random.default_rng(self.random_state)

        self.root_ = self._grow(X, encoded, depth=0)
        return self

    def _choose_features(self):
        """Select which features to consider at a node."""
        n = self.n_features_

        if self.max_features is None:
            count = n
        elif self.max_features == "sqrt":
            count = max(1, int(np.sqrt(n)))
        elif self.max_features == "log2":
            count = max(1, int(np.log2(n)))
        elif isinstance(self.max_features, int) and 1 <= self.max_features <= n:
            count = self.max_features
        else:
            raise ValueError("Invalid max_features")

        return self.rng_.choice(n, size=count, replace=False)

    def _best_split(self, X, y, parent_counts):
        """Find the feature and threshold with maximum information gain."""
        n = len(y)
        features = self._choose_features()

        # Representative thresholds instead of every unique midpoint
        q = np.linspace(0, 1, self.n_thresholds + 2)[1:-1]
        candidates = np.quantile(X[:, features], q, axis=0)

        parent_entropy = float(entropy(parent_counts))

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for j, feature in enumerate(features):
            thresholds = np.unique(candidates[:, j])

            # Group samples into bins for each candidate threshold
            bins = np.searchsorted(thresholds, X[:, feature], side="left")

            hist = np.bincount(
                bins * self.n_classes_ + y,
                minlength=(len(thresholds) + 1) * self.n_classes_,
            ).reshape(-1, self.n_classes_)

            left_counts = np.cumsum(hist, axis=0)[:-1]
            left_n = left_counts.sum(axis=1)
            right_n = n - left_n

            valid = (left_n >= self.min_samples_leaf) & (
                right_n >= self.min_samples_leaf
            )

            if not valid.any():
                continue

            right_counts = parent_counts - left_counts

            # Information gain = parent entropy - child entropy
            gains = parent_entropy - (
                (left_n / n) * entropy(left_counts)
                + (right_n / n) * entropy(right_counts)
            )

            gains[~valid] = -np.inf
            ix = int(np.argmax(gains))

            if gains[ix] > best_gain:
                best_gain = float(gains[ix])
                best_feature = int(feature)
                best_threshold = float(thresholds[ix])

        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        """Recursively construct the tree."""
        counts = np.bincount(y, minlength=self.n_classes_)

        # Majority class becomes the default prediction
        node = TreeNode(prediction=int(np.argmax(counts)))

        # Stopping conditions
        if (
            counts.max() == len(y)
            or len(y) < max(self.min_samples_split, 2 * self.min_samples_leaf)
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        feature, threshold = self._best_split(X, y, counts)

        if feature is None:
            return node

        mask = X[:, feature] <= threshold

        node.feature = feature
        node.threshold = threshold

        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)

        return node

    def predict(self, X):
        """Traverse the tree to predict class labels."""
        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit() before predict()")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError("Incorrect feature dimensions")

        encoded = np.empty(len(X), dtype=int)

        for i, row in enumerate(X):
            node = self.root_

            while node.feature != -1:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            encoded[i] = node.prediction

        return self.classes_[encoded]

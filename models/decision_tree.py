"""Multiclass decision tree with quantile-based split candidates.

The bins keep the split search manageable for 768-dimensional features.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class _Node:
    prediction: int
    probabilities: np.ndarray
    feature: int = -1
    threshold_bin: int = -1
    left: object = None
    right: object = None


class DecisionTree:
    def __init__(self, max_depth=12, min_samples_split=16,
                 min_samples_leaf=8, max_features=None, criterion="gini",
                 max_bins=32, min_impurity_decrease=1e-9, random_state=42):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if max_bins < 2 or max_bins > 256:
            raise ValueError("max_bins must be between 2 and 256")
        if min_samples_leaf < 1 or min_samples_split < 2:
            raise ValueError("invalid minimum sample settings")
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.criterion = criterion
        self.max_bins = int(max_bins)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state

    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or len(X) == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite two-dimensional array")
        return X

    @staticmethod
    def _make_bins(X, max_bins):
        # One common, data-driven set of thresholds per feature. The final bin
        # contains values greater than the final threshold. Ties are removed.
        quantiles = np.linspace(0.0, 1.0, max_bins + 1)[1:-1]
        edges = []
        binned = np.empty(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            candidates = np.unique(np.quantile(X[:, j], quantiles).astype(np.float32))
            # No split possible on a constant-valued column.
            candidates = candidates[candidates < np.max(X[:, j])]
            edges.append(candidates)
            binned[:, j] = np.searchsorted(candidates, X[:, j], side="left")
        return binned, edges

    def _num_features(self, n):
        v = self.max_features
        if v is None or v == "all":
            return n
        if v == "sqrt":
            return max(1, int(np.sqrt(n)))
        if v == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(v, float) and 0 < v <= 1:
            return max(1, int(np.ceil(v * n)))
        if isinstance(v, (int, np.integer)) and 1 <= v <= n:
            return int(v)
        raise ValueError("max_features must be None, sqrt, log2, fraction, or integer")

    def _impurity(self, counts):
        counts = np.asarray(counts, dtype=float)
        denom = np.sum(counts, axis=-1, keepdims=True)
        p = np.divide(counts, denom, out=np.zeros_like(counts), where=denom != 0)
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0), axis=-1)

    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one label per training row")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        binned, self.bin_edges_ = self._make_bins(X, self.max_bins)
        return self._fit_from_bins(binned, encoded, self.classes_, self.bin_edges_)

    def _fit_from_bins(self, binned, encoded, classes, edges, sample_indices=None):
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = binned.shape[1]
        self.bin_edges_ = edges
        self.n_classes_ = len(classes)
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        if sample_indices is None:
            sample_indices = np.arange(len(encoded), dtype=np.int32)
        self.root_ = self._grow(binned, np.asarray(encoded, dtype=np.int32),
                                np.asarray(sample_indices, dtype=np.int32), 0)
        norm = self.feature_importances_.sum()
        if norm > 0:
            self.feature_importances_ /= norm
        del self._rng
        return self

    def _grow(self, Xb, y, ids, depth):
        counts = np.bincount(y[ids], minlength=self.n_classes_)
        prob = counts / len(ids)
        node = _Node(int(np.argmax(counts)), prob)
        if (counts.max() == len(ids) or len(ids) < max(self.min_samples_split, 2*self.min_samples_leaf)
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        parent_imp = float(self._impurity(counts))
        features = self._rng.choice(self.n_features_in_,
                                    self._num_features(self.n_features_in_), replace=False)
        best_gain = self.min_impurity_decrease
        best_feature = -1
        best_threshold = -1
        n = len(ids)
        for j in features:
            n_bins = len(self.bin_edges_[j]) + 1
            if n_bins < 2:
                continue
            # Combine feature-bin and class into a single integer to build a
            # class histogram without Python loops over individual samples.
            frequencies = np.bincount(Xb[ids, j].astype(np.int32) * self.n_classes_ + y[ids],
                                      minlength=n_bins * self.n_classes_)
            histogram = frequencies.reshape(n_bins, self.n_classes_)
            left_counts = np.cumsum(histogram, axis=0)[:-1]
            left_sizes = left_counts.sum(axis=1)
            right_sizes = n - left_sizes
            valid = (left_sizes >= self.min_samples_leaf) & (right_sizes >= self.min_samples_leaf)
            if not valid.any():
                continue
            right_counts = counts - left_counts
            weighted = ((left_sizes / n) * self._impurity(left_counts)
                        + (right_sizes / n) * self._impurity(right_counts))
            gains = parent_imp - weighted
            gains[~valid] = -np.inf
            k = int(np.argmax(gains))
            if gains[k] > best_gain:
                best_gain = float(gains[k])
                best_feature, best_threshold = int(j), k
        if best_feature < 0:
            return node
        mask = Xb[ids, best_feature] <= best_threshold
        node.feature = best_feature
        node.threshold_bin = best_threshold
        self.feature_importances_[best_feature] += len(ids) * best_gain
        node.left = self._grow(Xb, y, ids[mask], depth + 1)
        node.right = self._grow(Xb, y, ids[~mask], depth + 1)
        return node

    def predict_proba(self, X):
        if not hasattr(self, "root_"):
            raise RuntimeError("fit the tree before prediction")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("wrong number of features")
        output = np.empty((len(X), self.n_classes_), dtype=float)
        def visit(node, idx):
            if len(idx) == 0:
                return
            if node.feature < 0:
                output[idx] = node.probabilities
                return
            threshold = self.bin_edges_[node.feature][node.threshold_bin]
            mask = X[idx, node.feature] <= threshold
            visit(node.left, idx[mask])
            visit(node.right, idx[~mask])
        visit(self.root_, np.arange(len(X)))
        return output

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# A descriptive alias for code that prefers explicit classifier names.
DecisionTreeClassifier = DecisionTree

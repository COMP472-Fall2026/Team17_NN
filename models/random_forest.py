import numpy as np

from models.decision_tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_trees=20,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features="sqrt",
        n_thresholds=8,
        random_state=42,
    ):
        if n_trees < 1:
            raise ValueError("n_trees must be positive")

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        """Train multiple trees on bootstrap samples."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError("Invalid training data")

        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []

        for _ in range(self.n_trees):
            # Sample training rows with replacement
            indices = rng.integers(0, len(y), size=len(y))

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_thresholds=self.n_thresholds,
                random_state=int(rng.integers(0, 2**31)),
            )

            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)

        return self

    def predict(self, X):
        """Predict using majority voting across all trees."""
        if not hasattr(self, "trees_"):
            raise RuntimeError("Call fit() before predict()")

        votes = np.stack([tree.predict(X) for tree in self.trees_], axis=0)

        vote_counts = np.zeros((votes.shape[1], len(self.classes_)), dtype=np.int32)

        for j, label in enumerate(self.classes_):
            vote_counts[:, j] = (votes == label).sum(axis=0)

        return self.classes_[np.argmax(vote_counts, axis=1)]

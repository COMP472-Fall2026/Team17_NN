"""Bootstrap ensemble of our own quantile-split DecisionTree implementation."""
import numpy as np
from models.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=40, max_depth=14, min_samples_split=16,
                 min_samples_leaf=5, max_features="sqrt", criterion="gini",
                 max_bins=32, bootstrap=True, max_samples=1.0,
                 random_state=42):
        if int(n_estimators) < 1:
            raise ValueError("n_estimators must be positive")
        if not 0 < max_samples <= 1:
            raise ValueError("max_samples must be in (0, 1]")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.max_bins = max_bins
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        X = DecisionTree._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("invalid labels")
        self.classes_, enc = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        rng = np.random.default_rng(self.random_state)
        bins, edges = DecisionTree._make_bins(X, self.max_bins)
        sample_size = max(1, int(round(len(X) * self.max_samples)))
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), size=sample_size, replace=self.bootstrap)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features,
                                criterion=self.criterion, max_bins=self.max_bins,
                                random_state=int(rng.integers(0, 2**31)))
            tree._fit_from_bins(bins, enc, self.classes_, edges, idx)
            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators
        return self

    def predict_proba(self, X):
        if not getattr(self, 'estimators_', None):
            raise RuntimeError("fit the forest before prediction")
        X = DecisionTree._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("wrong number of features")
        proba = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        for model in self.estimators_:
            proba += model.predict_proba(X)
        return proba / len(self.estimators_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

RandomForestClassifier = RandomForest

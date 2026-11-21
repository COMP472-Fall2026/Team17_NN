import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    FunctionTransformer,
)


class PowerScaler:
    def __init__(self, power=0.5, scaling_power=0.75):
        self.power = power
        self.scaling_power = scaling_power

    def fit(self, features, y=None):

        # Keep the sign while reducing the influence of large feature values
        features = np.asarray(features, dtype=np.float32)
        transformed = np.sign(features) * np.abs(features) ** self.power

        # Learn the scale from training data only; do not subtract the mean
        self.scale = transformed.std(axis=0) ** self.scaling_power
        self.scale[self.scale == 0] = 1
        return self

    def transform(self, features):

        features = np.asarray(features, dtype=np.float32)
        transformed = np.sign(features) * np.abs(features) ** self.power
        return transformed / self.scale

    def fit_transform(self, features, y=None):

        self.fit(features)
        return self.transform(features)


class GlobalScaler(BaseEstimator, TransformerMixin):
    """
    Centre each feature, then divide everything by ONE number
    (the overall standard deviation).

    Unlike StandardScaler, strong features stay strong and weak
    features stay weak.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = (X - self.mean_).std()
        if self.scale_ == 0:
            self.scale_ = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_


def signed_sqrt(X):
    """Square root of each value, keeping the sign of negative values."""
    X = np.asarray(X, dtype=np.float64)
    return np.sign(X) * np.sqrt(np.abs(X))


def make_preprocessor(method: str = "none"):
    """
    Create a preprocessing transformer.

    Scaling methods (keep every feature):
        none        - No transformation
        standard    - Standardize each feature
        l2          - Normalize each sample to unit length
        standard_l2 - Standardize, then L2 normalize
        minmax      - Scale features to [0, 1]
        partial     - Divide by training standard deviation raised to 0.75
        power       - Signed square root, then partial scaling
        global      - Centre each feature, divide by one overall std
        sqrt_l2     - Square root of each value, then L2 normalize
                      (best for k-NN on the sparse image features)
        sqrt_l2_global - sqrt_l2, then global scaling
                      (same idea, scaled for the neural network)

    Feature-reduction methods (N is any number, e.g. pca100, kbest200):
        pcaN        - Global scale, then compress to N PCA components
        kbestN      - Keep the N features that best separate the classes
                      (ANOVA F-test), then standardize them.
                      Needs the labels: fit_transform(X_train, y_train)

    Returns a transformer with fit_transform() and transform().
    Always fit on the training data only.
    """

    if method == "none":
        return FunctionTransformer()

    if method == "standard":
        return StandardScaler()

    if method == "l2":
        return Normalizer(norm="l2")

    if method == "standard_l2":
        return make_pipeline(StandardScaler(), Normalizer(norm="l2"))

    if method == "minmax":
        return MinMaxScaler()

    if method == "partial":
        return PowerScaler(power=1)

    if method == "power":
        return PowerScaler()

    if method == "global":
        return GlobalScaler()

    if method == "sqrt_l2":
        return make_pipeline(FunctionTransformer(signed_sqrt), Normalizer(norm="l2"))

    if method == "sqrt_l2_global":
        return make_pipeline(
            FunctionTransformer(signed_sqrt), Normalizer(norm="l2"), GlobalScaler()
        )

    match = re.fullmatch(r"pca(\d+)", method)
    if match:
        return make_pipeline(
            GlobalScaler(),
            PCA(n_components=int(match.group(1)), random_state=42),
            GlobalScaler(),  # bring the components back to unit scale
        )

    match = re.fullmatch(r"kbest(\d+)", method)
    if match:
        return make_pipeline(
            SelectKBest(f_classif, k=int(match.group(1))), StandardScaler()
        )

    raise ValueError(f"Unknown preprocessing method: {method}")

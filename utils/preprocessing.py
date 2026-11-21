import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    FunctionTransformer,
)


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

    Methods (keep every feature):
        none        - No transformation
        standard    - Standardize each feature
        l2          - Normalize each sample to unit length
        standard_l2 - Standardize, then L2 normalize
        minmax      - Scale features to [0, 1]
        sqrt_l2     - Square root of each value, then L2 normalize
                      (best for k-NN on the sparse image features)
        sqrt_l2_global - sqrt_l2, then global scaling
                      (same idea, scaled for the neural network)

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

    if method == "sqrt_l2":
        return make_pipeline(FunctionTransformer(signed_sqrt), Normalizer(norm="l2"))

    if method == "sqrt_l2_global":
        return make_pipeline(
            FunctionTransformer(signed_sqrt), Normalizer(norm="l2"), GlobalScaler()
        )

    raise ValueError(f"Unknown preprocessing method: {method}")

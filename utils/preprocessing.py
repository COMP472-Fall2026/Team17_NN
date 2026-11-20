
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    FunctionTransformer,
)
from sklearn.pipeline import make_pipeline


class PowerScaler:

    def __init__(self, power=0.5, scaling_power=0.75):
        self.power = power
        self.scaling_power = scaling_power

    def fit(self, features):

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

    def fit_transform(self, features):

        self.fit(features)
        return self.transform(features)


def make_preprocessor(method: str = "none"):
    """
    Create a preprocessing transformer.

    Methods:
        none        - No transformation
        standard    - Standardize each feature
        l2          - Normalize each sample to unit length
        standard_l2 - Standardize, then L2 normalize
        minmax      - Scale features to [0, 1]
        partial     - Divide by training standard deviation raised to 0.75
        power       - Signed square root, then partial scaling

    Returns a transformer with fit_transform() and transform().
    """

    if method == "none":
        return FunctionTransformer()

    if method == "standard":
        return StandardScaler()

    if method == "l2":
        return Normalizer(norm="l2")

    if method == "standard_l2":
        return make_pipeline(
            StandardScaler(),
            Normalizer(norm="l2")
        )

    if method == "minmax":
        return MinMaxScaler()

    if method == "partial":
        return PowerScaler(power=1)

    if method == "power":
        return PowerScaler()

    raise ValueError(f"Unknown preprocessing method: {method}")

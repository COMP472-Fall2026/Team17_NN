
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    Normalizer,
    FunctionTransformer,
)
from sklearn.pipeline import make_pipeline


def make_preprocessor(method: str = "none"):
    """
    Create a preprocessing transformer.

    Methods:
        none        - No transformation
        standard    - Standardize each feature
        l2          - Normalize each sample to unit length
        standard_l2 - Standardize, then L2 normalize
        minmax      - Scale features to [0, 1]

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

    raise ValueError(f"Unknown preprocessing method: {method}")

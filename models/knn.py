import numpy as np


class KNN:
    def __init__(
        self,
        k=5,
        distance="euclidean",
        batch_size=128,
        density_penalty=0,
        density_neighbors=20,
    ):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        if distance not in ("euclidean", "cosine"):
            raise ValueError("distance must be 'euclidean' or 'cosine'")

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(density_penalty) or density_penalty < 0:
            raise ValueError("density_penalty must be nonnegative")
        if not isinstance(density_neighbors, int) or density_neighbors < 1:
            raise ValueError("density_neighbors must be a positive integer")
        if density_penalty > 0 and distance != "cosine":
            raise ValueError("Density correction requires cosine distance")

        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.density_penalty = density_penalty
        self.density_neighbors = density_neighbors

    def fit(self, X, y):
        """Store training examples and prepare distance calculations."""

        self.X_train = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.X_train.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if len(self.X_train) != len(y) or self.k > len(y):
            raise ValueError("Invalid training data or k")

        self.classes_, self.y_encoded = np.unique(y, return_inverse=True)

        if self.distance == "euclidean":
            self.train_sq_norms = np.sum(self.X_train**2, axis=1)
        else:
            norms = np.linalg.norm(self.X_train, axis=1, keepdims=True)
            self.X_normalized = self.X_train / np.maximum(norms, 1e-12)

            if self.density_penalty > 0:
                self._fit_density()

        return self

    def _fit_density(self):

        if len(self.X_train) < 2:
            raise ValueError("Density correction needs at least two training examples")

        # Measure how similar each example is to other training examples
        count = min(self.density_neighbors, len(self.X_train) - 1)
        self.train_density = np.empty(len(self.X_train), dtype=np.float32)

        for start in range(0, len(self.X_train), self.batch_size):
            batch = self.X_normalized[start : start + self.batch_size]
            similarities = batch @ self.X_normalized.T

            # Do not count an example as its own neighbor
            rows = np.arange(len(batch))
            similarities[rows, start + rows] = -np.inf

            nearest = np.partition(similarities, len(self.X_train) - count, axis=1)[
                :, -count:
            ]
            self.train_density[start : start + len(batch)] = nearest.mean(axis=1)

    def predict(self, X):
        """Predict class labels using k nearest neighbours."""

        if not hasattr(self, "X_train"):
            raise RuntimeError("Call fit() before predict()")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Invalid feature dimensions")

        predictions = []

        for start in range(0, len(X), self.batch_size):
            batch = X[start : start + self.batch_size]

            if self.distance == "euclidean":
                batch_sq_norms = np.sum(batch**2, axis=1, keepdims=True)

                distances = (
                    batch_sq_norms
                    + self.train_sq_norms[None, :]
                    - 2 * batch @ self.X_train.T
                )

                nearest = np.argpartition(distances, self.k - 1, axis=1)[:, : self.k]

            else:
                norms = np.linalg.norm(batch, axis=1, keepdims=True)
                normalized = batch / np.maximum(norms, 1e-12)

                similarities = normalized @ self.X_normalized.T

                # Reduce the advantage of training examples that match many queries
                if self.density_penalty > 0:
                    similarities -= self.density_penalty * self.train_density

                nearest = np.argpartition(-similarities, self.k - 1, axis=1)[
                    :, : self.k
                ]

            for indices in nearest:
                votes = self.y_encoded[indices]
                counts = np.bincount(votes, minlength=len(self.classes_))

                predictions.append(self.classes_[np.argmax(counts)])

        return np.asarray(predictions)

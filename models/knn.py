
import numpy as np


class KNN:
    def __init__(self, k=5, distance="euclidean", batch_size=128):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        if distance not in ("euclidean", "cosine"):
            raise ValueError("distance must be 'euclidean' or 'cosine'")

        self.k = k
        self.distance = distance
        self.batch_size = batch_size

    def fit(self, X, y):
        """Store training examples and prepare distance calculations."""

        self.X_train = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if self.X_train.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D")

        if len(self.X_train) != len(y) or self.k > len(y):
            raise ValueError("Invalid training data or k")

        self.classes_, self.y_encoded = np.unique(
            y, return_inverse=True
        )

        if self.distance == "euclidean":
            self.train_sq_norms = np.sum(
                self.X_train ** 2, axis=1
            )
        else:
            norms = np.linalg.norm(
                self.X_train, axis=1, keepdims=True
            )
            self.X_normalized = self.X_train / np.maximum(
                norms, 1e-12
            )

        return self

    def predict(self, X):
        """Predict class labels using k nearest neighbours."""

        if not hasattr(self, "X_train"):
            raise RuntimeError("Call fit() before predict()")

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2 or X.shape[1] != self.X_train.shape[1]:
            raise ValueError("Invalid feature dimensions")

        predictions = []

        for start in range(0, len(X), self.batch_size):
            batch = X[start:start + self.batch_size]

            if self.distance == "euclidean":
                batch_sq_norms = np.sum(
                    batch ** 2, axis=1, keepdims=True
                )

                distances = (
                    batch_sq_norms
                    + self.train_sq_norms[None, :]
                    - 2 * batch @ self.X_train.T
                )

                nearest = np.argpartition(
                    distances, self.k - 1, axis=1
                )[:, :self.k]

            else:
                norms = np.linalg.norm(
                    batch, axis=1, keepdims=True
                )
                normalized = batch / np.maximum(norms, 1e-12)

                similarities = normalized @ self.X_normalized.T

                nearest = np.argpartition(
                    -similarities, self.k - 1, axis=1
                )[:, :self.k]

            for indices in nearest:
                votes = self.y_encoded[indices]
                counts = np.bincount(
                    votes, minlength=len(self.classes_)
                )

                predictions.append(
                    self.classes_[np.argmax(counts)]
                )

        return np.asarray(predictions)
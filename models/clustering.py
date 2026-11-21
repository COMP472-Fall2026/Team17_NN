import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, n_init=5, seed=42):
        if n_clusters < 1 or max_iters < 1 or n_init < 1:
            raise ValueError("Clusters, iterations, and initializations must be positive")

        # Number of buckets, rounds per attempt, and different starting attempts
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.n_init = n_init
        self.seed = seed

    def _squared_distances(self, features, centers):
        """Calculate squared distances to the cluster centers."""

        # Squared distance = sum(a^2) + sum(b^2) - 2 * dot(a, b)
        # Each row contains one example's distances to all centers
        sample_lengths = np.sum(features ** 2, axis=1, keepdims=True)
        center_lengths = np.sum(centers ** 2, axis=1)
        distances = sample_lengths + center_lengths - 2 * features @ centers.T

        # Remove tiny negative values caused by rounding
        return np.maximum(distances, 0)

    def fit(self, features):
        """Find clusters using features without labels."""

        # Sentiment labels are never passed to this method
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] == 0 or len(features) < self.n_clusters:
            raise ValueError("Provide a 2D feature array with at least one example per cluster")
        if not np.isfinite(features).all():
            raise ValueError("Features must contain finite numbers")

        # The seed makes the starting points repeatable
        rng = np.random.default_rng(self.seed)
        best_inertia = np.inf

        # Run several attempts and keep the most compact grouping
        for run in range(self.n_init):
            # Start with randomly selected examples as centers
            indices = rng.choice(len(features), self.n_clusters, replace=False)
            centers = features[indices].copy()

            for iteration in range(self.max_iters):
                # Assign every example to the closest center
                distances = self._squared_distances(features, centers)
                clusters = np.argmin(distances, axis=1)

                # Move each center to the average of its assigned examples
                new_centers = centers.copy()
                for cluster in range(self.n_clusters):
                    members = features[clusters == cluster]
                    if len(members) > 0:
                        new_centers[cluster] = members.mean(axis=0)
                    else:
                        # Restart an empty bucket at a faraway example
                        farthest = np.argmax(distances.min(axis=1))
                        new_centers[cluster] = features[farthest]

                # Stop when the centers barely move
                movement = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if movement < 0.0001:
                    break

            # Reassign examples to the final centers before measuring this run
            distances = self._squared_distances(features, centers)
            clusters = np.argmin(distances, axis=1)
            inertia = distances[np.arange(len(features)), clusters].sum()

            # Smaller inertia means examples are closer to their assigned centers
            if inertia < best_inertia:
                best_inertia = inertia
                self.centers = centers.copy()
                self.labels = clusters.copy()  # Bucket numbers, not sentiment labels
                self.inertia = float(inertia)
                self.iterations = iteration + 1

        return self

    def predict(self, features):
        """Assign examples to the closest cluster center."""

        if not hasattr(self, "centers"):
            raise ValueError("Call fit before predict")

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.centers.shape[1]:
            raise ValueError("Features must match the fitted feature columns")
        if not np.isfinite(features).all():
            raise ValueError("Features must contain finite numbers")

        # Use the existing centers without moving them
        distances = self._squared_distances(features, self.centers)
        return np.argmin(distances, axis=1)

    def fit_predict(self, features):
        """Find clusters and return their group numbers."""

        self.fit(features)
        return self.labels.copy()

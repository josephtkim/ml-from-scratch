import numpy as np

class KMeans:
    def __init__(self, n_clusters: int, max_iters: int = 100, tol: float = 1e-4):
        """
        Initialize number of clusters and convergence criteria.
        Args:
            n_clusters (int): Number of clusters (K).
            max_iters (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def initialize_centroids(self, X: np.ndarray):
        """
        Randomly initialize centroids from data.
        Args:
            X (np.ndarray): Data points of shape (n_samples, n_features).
        """
        n_samples = X.shape[0]
        
        # Pick K random, unique indices 
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # Select the points as initial centroids 
        self.centroids = X[random_indices, :]

    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the closest centroid.
        Args:
            X (np.ndarray): Data points.
        Returns:
            np.ndarray: Cluster assignments (shape: n_samples,)
        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_centroids(self, X: np.ndarray, labels: np.ndarray):
        """
        Update centroids as the mean of assigned points.
        Handles empty clusters by re-seeding that centroid to a random data point.
        Args:
            X (np.ndarray): Data points, shape (n_samples, n_features).
            labels (np.ndarray): Cluster assignments, shape (n_samples,).
        Returns:
            float: L2 shift of centroids (useful for convergence checks).
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features), dtype=X.dtype)
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
                
        if self.centroids is None:
            shift = np.inf
        else:
            shift = np.linalg.norm(self.centroids - new_centroids)
            
        self.centroids = new_centroids 
        return shift

    def compute_loss(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute total intra-cluster variance (loss).
        Args:
            X (np.ndarray): Data points.
            labels (np.ndarray): Cluster assignments.
        Returns:
            float: Total distortion/loss.
        """
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Call initialize_centroids() or fit() first.")
            
        labels = labels.ravel()
        assigned = self.centroids[labels]
        
        diff = X - assigned 
        loss = np.sum(diff * diff)
        
        return float(loss)

    def fit(self, X: np.ndarray):
        """
        Run the K-Means algorithm:
          1) Initialize centroids (if not already set)
          2) Repeat until convergence or max_iters:
             - Assign each point to nearest centroid
             - Update centroids as mean of assigned points
             - Check centroid shift against tolerance
          3) Optionally compute final loss
        Stores:
          - self.centroids: final centroids
          - self.inertia_: final distortion (sum of squared distances)
          - self.n_iter_: number of iterations actually run
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if self.centroids is None:
            self.initialize_centroids(X)

        # Optional trackers
        self.n_iter_ = 0
        self.inertia_ = None

        # 1) Main loop
        for it in range(self.max_iters):
            # (a) Assignment step
            labels = self.assign_clusters(X)

            # (b) Update step (returns centroid shift for convergence)
            shift = self.update_centroids(X, labels)
            self.n_iter_ = it + 1

            # (c) Convergence check
            if np.isfinite(shift) and shift <= self.tol:
                break

        # 2) Final assignments & loss (in case last update moved centroids)
        self.labels_ = self.assign_clusters(X)
        self.labels = self.labels_
        self.inertia_ = self.compute_loss(X, self.labels)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict closest cluster for each data point (after fit).
        Args:
            X (np.ndarray): Data points.
        Returns:
            np.ndarray: Predicted cluster indices.
        """
        if self.centroids is None:
            raise ValueError("Centroids are not initialized. Fit the model or call initialize_centroids() first.")

        return self.assign_clusters(X)
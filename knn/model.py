import numpy as np

class KNearestNeighbors:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN classifier.
        Args:
            k (int): Number of neighbors to consider.
            distance_metric (str): Metric to use for distance computation.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data (no learning).
        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training labels, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array (n_samples,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        
        self.X_train = X
        self.y_train = y 
        
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} cannot exceed number of training samples ({X.shape[0]}).")
            
        return self

    def compute_distances(self, X_test):
        """
        Compute distances between test samples and all training samples.
        Args:
            X_test (np.ndarray): Test features, shape (m_samples, n_features)
        Returns:
            distances (np.ndarray): shape (m_samples, n_train_samples)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("fit was not called before compute_distances().")
            
        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError("X_test must be 2D (m_samples, n_features).")
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_test has {X_test.shape[1]}, "
                f"but X_train has {self.X_train.shape[1]}."
            )
            
        if self.distance_metric == "euclidean":
            # Efficient pairwise L2:
            # D^2 = ||X_test||^2 + ||X_train||^2 - 2 X_test X_train^T 
            X_test_sq = np.sum(X_test**2, axis=1, keepdims=True)          # (m,1)
            X_train_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T # (1,n)
            # (m,1) + (1,n) -> (m,n)
            d2 = X_test_sq + X_train_sq - 2.0 * (X_test @ self.X_train.T) # (m,n)
            d2 = np.maximum(d2, 0.0)
            distances = np.sqrt(d2)                                       # (m,n)
        elif self.distance_metric == "manhattan":
            # L1: sum over features of |x_i - y_j|
            # Broadcast to (m,n,f) then reduce over features 
            distances = np.sum(
                np.abs(X_test[:, None, :] - self.X_train[None, :, :]),
                axis=2
            ) # (m,n)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances

    def predict(self, X_test):
        """
        Predict class labels for test data.
        Args:
            X_test (np.ndarray): Test features.
        Returns:
            np.ndarray: Predicted labels, shape (m_samples,)
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("KNearestNeighbors not fitted. Call fit(X, y) before predict().")
        
        distances = self.compute_distances(X_test)
        m_samples = distances.shape[0]
        predictions = np.empty((m_samples,), dtype=self.y_train.dtype)
        
        for i in range(m_samples):
            # 1) indices of k smallest distances for sample i 
            knn_idx = np.argpartition(distances[i], self.k - 1)[:self.k]
            
            # 2) neighbor labels (and distances, since you use them for tie-break)
            neighbor_labels = self.y_train[knn_idx] 
            neighbor_dists = distances[i][knn_idx]
            
            # 3) frequency per label 
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            tied_labels = labels[counts == max_count]
            
            if tied_labels.size == 1:
                predictions[i] = tied_labels[0]
            else:
                # 4/5) tie-break: smallest total distance among tied labels, then smallest label 
                best_label = None
                best_sum = np.inf 
                for lbl in tied_labels:
                    s = neighbor_dists[neighbor_labels == lbl].sum()
                    if (s < best_sum) or (s == best_sum and (best_label is None or lbl < best_label)):
                        best_sum = s 
                        best_label = lbl 
                predictions[i] = best_label 
                
        return predictions        
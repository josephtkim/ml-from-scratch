import numpy as np

class PCA:
    def __init__(self, n_components: int):
        """
        Initialize PCA with number of components to keep.
        Args:
            n_components (int): Number of principal components (k).
        """
        self.n_components = n_components
        self.components = None  # shape: (d, k)
        self.mean = None        # shape: (d,)

    def fit(self, X: np.ndarray):
        """
        Fit PCA model to data X.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).
        """
        # Step 1: Center the data 
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean 
        
        # Step 2: Compute covariance matrix 
        cov = np.cov(X_centered, rowvar=False)
        
        # Step 3: Eigen-decomposition of covariance matrix 
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order 
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvecs_sorted = eigvecs[:, sorted_idx]
        
        # Step 5: Select top-k eigenvectors (principal components)
        self.components = eigvecs_sorted[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Reduced data of shape (n_samples, k).
        """
        if self.mean is None or self.components is None:
            raise ValueError("PCA model has not been fitted yet.")
            
        # Step 1: Center input data using the stored mean 
        X_centered = X - self.mean 
        
        # Step 2: Project onto principal components 
        X_reduced = X_centered @ self.components 
        
        return X_reduced 

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from reduced representation.
        Args:
            X_reduced (np.ndarray): Reduced data.
        Returns:
            np.ndarray: Reconstructed data of shape (n_samples, n_features).
        """
        if self.components is None or self.mean is None:
            raise ValueError("PCA model has not been fitted yet.")
        
        # Step 1: Project reduced data back to original space 
        X_reconstructed = X_reduced @ self.components.T 
        
        # Step 2: Add back the mean 
        X_reconstructed += self.mean 
        
        return X_reconstructed
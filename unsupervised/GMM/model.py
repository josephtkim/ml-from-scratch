import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureModel:
    def __init__(self, n_components: int, max_iters: int = 100, tol: float = 1e-4):
        """
        Initialize parameters for GMM.
        Args:
            n_components (int): Number of Gaussian clusters.
            max_iters (int): Maximum iterations for EM.
            tol (float): Convergence tolerance.
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol

        # Parameters to learn
        self.weights = None   # π_k
        self.means = None     # μ_k
        self.covariances = None  # Σ_k

    def initialize_parameters(self, X: np.ndarray):
        """
        Randomly initialize means, covariances, and mixing coefficients.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        K = self.n_components

        # Initialize means by randomly selecting K data points
        indices = np.random.choice(n_samples, K, replace=False)
        self.means = X[indices]

        # Initialize covariances to identity matrices
        self.covariances = np.array([np.eye(n_features) for _ in range(K)])

        # Initialize weights uniformly (can also use Dirichlet for random variation)
        self.weights = np.full(K, 1.0 / K)

    def e_step(self, X: np.ndarray) -> np.ndarray:
        """
        Perform the E-step: compute responsibilities γ_{ik}.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        Returns:
            responsibilities (np.ndarray): shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        K = self.n_components

        responsibilities = np.zeros((n_samples, K))

        for k in range(K):
            # Gaussian density for component k
            rv = multivariate_normal(mean=self.means[k], cov=self.covariances[k], allow_singular=True)
            
            # Numerator: π_k * N(x_i | μ_k, Σ_k)
            responsibilities[:, k] = self.weights[k] * rv.pdf(X)

        # Normalize over all components (rows sum to 1)
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= responsibilities_sum  # shape: (n_samples, K)

        return responsibilities

    def m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """
        Perform the M-step: update weights, means, and covariances.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
            responsibilities (np.ndarray): shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        K = self.n_components

        # Initialize updated parameters
        self.weights = np.zeros(K)
        self.means = np.zeros((K, n_features))
        self.covariances = np.zeros((K, n_features, n_features))

        for k in range(K):
            # Responsibilities for component k
            gamma_k = responsibilities[:, k]  # shape: (n_samples,)

            # Effective number of points assigned to cluster k
            N_k = np.sum(gamma_k)

            # Update mean
            self.means[k] = np.sum(gamma_k[:, np.newaxis] * X, axis=0) / N_k

            # Update covariance
            diff = X - self.means[k]  # shape: (n_samples, n_features)
            weighted_outer = np.einsum('i,ij,ik->jk', gamma_k, diff, diff)
            self.covariances[k] = weighted_outer / N_k

            # Update mixing coefficient
            self.weights[k] = N_k / n_samples

    def compute_log_likelihood(self, X: np.ndarray) -> float:
        """
        Compute total log-likelihood of the data under the current model.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        Returns:
            float: Log-likelihood value
        """
        n_samples = X.shape[0]
        K = self.n_components

        log_likelihood = 0.0
        for i in range(n_samples):
            prob = 0.0
            for k in range(K):
                rv = multivariate_normal(mean=self.means[k],
                                         cov=self.covariances[k],
                                         allow_singular=True)
                prob += self.weights[k] * rv.pdf(X[i])
            log_likelihood += np.log(prob + 1e-12)  # small epsilon to avoid log(0)
        return log_likelihood

    def fit(self, X: np.ndarray, return_history: bool = False):
        """
        Run the EM algorithm until convergence.
        Args:
            X (np.ndarray): Input data
            return_history (bool): Whether to return per-iteration log-likelihoods
        Returns:
            Optional: list of log-likelihoods if return_history=True
        """
        prev_log_likelihood = None
        history = []

        for iteration in range(self.max_iters):
            # E-step
            responsibilities = self.e_step(X)

            # M-step
            self.m_step(X, responsibilities)

            # Compute current log-likelihood
            log_likelihood = self.compute_log_likelihood(X)
            history.append(log_likelihood)

            # Check for convergence
            if prev_log_likelihood is not None:
                if abs(log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Converged at iteration {iteration}")
                    break
            prev_log_likelihood = log_likelihood

        if return_history:
            return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute responsibilities for each data point.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        Returns:
            responsibilities (np.ndarray): shape (n_samples, n_components)
        """
        return self.e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely cluster (hard assignment) for each sample.
        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features)
        Returns:
            np.ndarray: Cluster labels of shape (n_samples,)
        """
        responsibilities = self.predict_proba(X)
        return np.argmax(responsibilities, axis=1)

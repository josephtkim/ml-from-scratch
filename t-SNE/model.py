import numpy as np
from scipy.spatial.distance import pdist, squareform

class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, lr=200.0, n_iters=1000):
        """
        Initialize t-SNE parameters.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iters = n_iters

    def _compute_pairwise_affinities(self, X):
        """
        Compute high-dimensional similarities P_{ij} using a Gaussian kernel.
        Returns:
            P (np.ndarray): Symmetric affinity matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        distances = squareform(pdist(X, 'sqeuclidean'))  # shape: (n, n)
        
        # Binary search for sigmas to match target perplexity
        def binary_search_sigma(dist_row, target_perplexity=30.0, tol=1e-5, max_iter=50):
            beta = 1.0  # = 1 / (2 * sigma^2)
            betamin, betamax = None, None

            for _ in range(max_iter):
                P = np.exp(-dist_row * beta)
                P[dist_row == 0] = 0  # zero self-similarity
                sumP = np.sum(P)
                if sumP == 0:
                    sumP = 1e-12
                P /= sumP
                entropy = -np.sum(P * np.log2(P + 1e-12))
                perp = 2 ** entropy

                if np.abs(perp - target_perplexity) < tol:
                    break

                if perp > target_perplexity:
                    betamin = beta
                    beta = beta * 2 if betamax is None else (beta + betamax) / 2
                else:
                    betamax = beta
                    beta = beta / 2 if betamin is None else (beta + betamin) / 2

            return beta

        # Compute conditional p_{j|i}
        P = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            beta = binary_search_sigma(distances[i], self.perplexity)
            row = np.exp(-distances[i] * beta)
            row[i] = 0  # no self-similarity
            row /= np.sum(row)
            P[i, :] = row

        # Symmetrize
        P = (P + P.T) / (2 * n_samples)
        return P

    def _compute_low_dim_affinities(self, Y):
        """
        Compute low-dimensional similarities Q_{ij} using Student's t-distribution.
        Returns:
            Q (np.ndarray): Affinity matrix of shape (n_samples, n_samples)
        """
        n_samples = Y.shape[0]
        distances = squareform(pdist(Y, 'sqeuclidean'))  # shape: (n, n)

        # t-distribution kernel
        inv_distances = 1 / (1 + distances)
        np.fill_diagonal(inv_distances, 0.0)  # zero out self-similarity

        Q = inv_distances / np.sum(inv_distances)
        return Q

    def _kl_divergence(self, P, Q):
        """
        Compute KL divergence between P and Q.
        """
        eps = 1e-12  # to avoid log(0)
        kl = np.sum(P * np.log((P + eps) / (Q + eps)))
        return kl

    def _compute_gradients(self, P, Q, Y):
        """
        Compute gradients of the KL divergence w.r.t. Y.
        Args:
            P (np.ndarray): High-dim similarity matrix (n x n)
            Q (np.ndarray): Low-dim similarity matrix (n x n)
            Y (np.ndarray): Low-dim coordinates (n x 2)
        Returns:
            dY (np.ndarray): Gradient w.r.t Y
        """
        n = Y.shape[0]
        dY = np.zeros_like(Y)  # shape: (n, 2)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                diff = Y[i] - Y[j]  # shape: (2,)
                dist_sq = np.sum(diff ** 2)
                grad_coeff = 4 * (P[i, j] - Q[i, j]) * (1 + dist_sq) ** -1
                dY[i] += grad_coeff * diff

        return dY

    def fit_transform(self, X):
        """
        Run t-SNE on input X and return 2D embedding.
        Args:
            X (np.ndarray): shape (n_samples, input_dim)
        Returns:
            Y (np.ndarray): shape (n_samples, n_components)
        """
        n_samples = X.shape[0]

        # Step 1: Initialize Y randomly
        Y = np.random.randn(n_samples, self.n_components) * 1e-4  # small init

        # Step 2: Compute high-dim affinities
        P = self._compute_pairwise_affinities(X)

        for it in range(self.n_iters):
            # Step 3: Compute low-dim affinities
            Q = self._compute_low_dim_affinities(Y)

            # Step 4: Compute KL divergence (for monitoring)
            loss = self._kl_divergence(P, Q)

            # Step 5: Compute gradient
            dY = self._compute_gradients(P, Q, Y)

            # Step 6: Gradient descent step
            Y -= self.lr * dY

            # Optional logging
            if it % 100 == 0 or it == self.n_iters - 1:
                print(f"Iteration {it}, KL Divergence: {loss:.4f}")

        return Y

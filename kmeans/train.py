import numpy as np
from model import KMeans

def generate_dummy_clusters(n_samples: int, n_features: int, n_clusters: int):
    """
    Generate synthetic Gaussian blobs for clustering.
    Returns:
        X (np.ndarray): Data points of shape (n_samples, n_features).
    """
    if n_samples < n_clusters:
        raise ValueError("n_samples must be >= n_clusters")

    rng = np.random.default_rng(42)

    # Allocate samples per cluster as evenly as possible
    base = n_samples // n_clusters
    rem = n_samples % n_clusters
    sizes = [base + (1 if i < rem else 0) for i in range(n_clusters)]

    # Randomly place cluster centers in a wider box to avoid overlap
    centers = rng.uniform(low=-8.0, high=8.0, size=(n_clusters, n_features))

    # Per-cluster isotropic scales (spread)
    scales = rng.uniform(low=0.5, high=1.5, size=n_clusters)

    blobs = []
    for k, (ck, sk, nk) in enumerate(zip(centers, scales, sizes)):
        # Draw nk points ~ N(ck, (sk^2) I)
        noise = rng.normal(loc=0.0, scale=sk, size=(nk, n_features))
        blobs.append(ck + noise)

    X = np.vstack(blobs).astype(np.float64)
    return X

def train(model: KMeans, X: np.ndarray, visualize: bool = False):
    """
    Fit the KMeans model and optionally print/visualize results.
    Args:
        model (KMeans): The clustering model.
        X (np.ndarray): Data points, shape (n_samples, n_features).
        visualize (bool): If True and n_features == 2, plot clusters & centroids.
    Returns:
        KMeans: The fitted model (with labels_, inertia_, centroids, n_iter_).
    """
    # Run K-Means
    model.fit(X)

    # Report final metrics
    print(f"Converged in {getattr(model, 'n_iter_', 'NA')} iterations.")
    print(f"Final loss (inertia): {getattr(model, 'inertia_', float('nan')):.6f}")

    # Optional 2D visualization
    if visualize and X.shape[1] == 2:
        try:
            import matplotlib.pyplot as plt
            labels = getattr(model, 'labels_', None)
            centroids = getattr(model, 'centroids', None)

            if labels is None or centroids is None:
                # Fallback: compute labels if not stored
                labels = model.predict(X)

            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=16, alpha=0.8)
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=128)
            plt.title("K-Means Clusters")
            plt.xlabel("x₁")
            plt.ylabel("x₂")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"(Visualization skipped: {e})")

    return model

if __name__ == "__main__":
    n_samples = 300
    n_features = 2
    n_clusters = 4

    # Generate data
    X = generate_dummy_clusters(n_samples, n_features, n_clusters)

    # Initialize model
    model = KMeans(n_clusters=n_clusters, max_iters=100)

    # Train model
    train(model, X, True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from model import GaussianMixtureModel

def generate_dummy_data(n_samples: int, n_features: int, n_components: int, seed: int = 42):
    """
    Generate synthetic data from a mixture of Gaussians.
    Returns:
        X (np.ndarray): Samples of shape (n_samples, n_features).
    """
    np.random.seed(seed)
    X = []

    for k in range(n_components):
        mean = np.random.randn(n_features) * 5
        cov = np.eye(n_features) * (0.5 + np.random.rand())  # random scaling
        samples = np.random.multivariate_normal(mean, cov, size=n_samples // n_components)
        X.append(samples)

    X = np.vstack(X)
    np.random.shuffle(X)
    return X

def train_gmm(X: np.ndarray, n_components: int, max_iters: int = 100):
    """
    Train a Gaussian Mixture Model and return model + training history.
    """
    model = GaussianMixtureModel(n_components=n_components, max_iters=max_iters)
    model.initialize_parameters(X)
    history = model.fit(X, return_history=True)
    return model, history

def plot_gmm(X, means, covariances, labels, title="GMM Clustering", filename=None):
    """
    Plot 2D GMM clustering results with ellipses.
    """
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Scatter data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, alpha=0.6)

    # Plot ellipses
    for mean, cov in zip(means, covariances):
        if cov.shape != (2, 2):
            continue  # skip non-2D
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals)  # 1 std dev

        ellipse = Ellipse(xy=mean, width=width, height=height,
                          angle=angle, edgecolor='black', fc='none', lw=2)
        ax.add_patch(ellipse)

    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    
def plot_log_likelihood(log_likelihoods, filename=None):
    """
    Plot log-likelihood over EM iterations.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(log_likelihoods, marker='o')
    plt.title("Log-Likelihood Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    n_samples = 500
    n_features = 2
    n_components = 3

    # Generate synthetic dataset
    X = generate_dummy_data(n_samples, n_features, n_components)

    # Train GMM and get history
    gmm, history = train_gmm(X, n_components=n_components, max_iters=100)

    # Predict cluster assignments
    labels = gmm.predict(X)
    means = gmm.means
    covariances = gmm.covariances

    # Plot GMM clustering result
    plot_gmm(X, means, covariances, labels)

    # Plot log-likelihood curve
    plot_log_likelihood(history)

    # Print summary
    print(f"Final log-likelihood: {history[-1]:.4f}")
    print(f"Mixing coefficients: {np.round(gmm.weights, 3)}")

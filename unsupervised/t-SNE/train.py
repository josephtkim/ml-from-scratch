import numpy as np
import matplotlib.pyplot as plt
from model import TSNE
from sklearn.datasets import load_iris  # or replace with make_blobs, MNIST, etc.

def load_data():
    """
    Load or generate high-dimensional data.
    Returns:
        X (np.ndarray): shape (n_samples, n_features)
        labels (np.ndarray): shape (n_samples,)
    """
    data = load_iris()
    X = data.data          # shape: (150, 4)
    labels = data.target   # shape: (150,)
    return X, labels

def plot_embedding(Y: np.ndarray, labels: np.ndarray):
    """
    Visualize the 2D embeddings with matplotlib.
    Args:
        Y (np.ndarray): 2D coordinates from t-SNE, shape (n_samples, 2)
        labels (np.ndarray): class labels for coloring
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load high-dimensional data
    X, labels = load_data()

    # Initialize t-SNE model
    model = TSNE(
        n_components=2,
        perplexity=30.0,
        lr=200.0,
        n_iters=2000
    )

    # Fit model and reduce to 2D
    Y = model.fit_transform(X)

    # Visualize the result
    plot_embedding(Y, labels)

import numpy as np
from model import PCA

def generate_dummy_data(n_samples: int, input_dim: int, noise_std: float = 0.1):
    """
    Generate synthetic data with some linear correlation.
    Returns:
        X (np.ndarray): Data matrix.
    """
    np.random.seed(42)

    # Step 1: Create low-rank latent space (true source signals)
    k_true = min(3, input_dim)  # number of true latent components
    Z = np.random.randn(n_samples, k_true)  # shape: (n, k)

    # Step 2: Random linear transformation to higher-dim space
    W = np.random.randn(k_true, input_dim)  # shape: (k, d)
    X_clean = Z @ W  # shape: (n, d)

    # Step 3: Add Gaussian noise
    noise = noise_std * np.random.randn(n_samples, input_dim)
    X_noisy = X_clean + noise

    return X_noisy

def train():
    # === Settings ===
    n_samples = 500
    input_dim = 5
    n_components = 2

    # === Step 1: Generate data ===
    X = generate_dummy_data(n_samples, input_dim)

    # === Step 2: Initialize and fit PCA ===
    model = PCA(n_components=n_components)
    model.fit(X)

    # === Step 3: Transform to low-dimensional space ===
    X_reduced = model.transform(X)

    # === Step 4: Inverse transform (reconstruct) ===
    X_reconstructed = model.inverse_transform(X_reduced)

    # === Step 5: Evaluate reconstruction error ===
    mse = np.mean((X - X_reconstructed) ** 2)

    # === Step 6: Print results ===
    print(f"Original dim: {X.shape[1]}")
    print(f"Reduced dim:  {X_reduced.shape[1]}")
    print(f"Reconstruction MSE: {mse:.6f}")

if __name__ == "__main__":
    train()

import numpy as np
from model import Autoencoder

def generate_dummy_data(n_samples: int, input_dim: int, hidden_dim: int, noise_std: float = 0.1, seed: int = 42):
    """
    Generate structured low-rank data with optional noise.
    The true dimensionality is hidden_dim (simulates latent structure).
    """
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n_samples, hidden_dim))        # Latent codes
    V = rng.standard_normal((hidden_dim, input_dim))        # Decoder matrix
    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_samples, input_dim))
    X = U @ V + noise
    return X.astype(np.float32)

def train(model, X, epochs=100, lr=0.01, reduction: str = "mean"):
    """
    Train the autoencoder model.
    """
    for epoch in range(epochs):
        Z = model.encode(X)
        X_hat = model.decode(Z)
        loss = model.compute_loss(X, X_hat, reduction=reduction)
        grads = model.compute_gradients(X, reduction=reduction)
        model.update_params(grads, lr)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss ({reduction}): {loss:.6f}")

def inspect_reconstructions(X, X_hat, n_samples: int = 3):
    """
    Print original and reconstructed vectors for visual comparison.
    """
    print("\n🔍 Reconstruction Comparison")
    for i in range(n_samples):
        print(f"\nSample {i}")
        print("Original:     ", np.round(X[i], 2))
        print("Reconstructed:", np.round(X_hat[i], 2))

if __name__ == "__main__":
    input_dim = 4
    hidden_dim = 10
    n_samples = 200

    # Generate structured data
    X = generate_dummy_data(n_samples, input_dim, hidden_dim)

    # Initialize model
    model = Autoencoder(input_dim, hidden_dim)

    # Train model
    train(model, X, epochs=5000, lr=0.01)

    # Inspect some reconstructions
    Z = model.encode(X)
    X_hat = model.decode(Z)
    inspect_reconstructions(X, X_hat, n_samples=3)

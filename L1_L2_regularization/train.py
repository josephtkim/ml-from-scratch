import numpy as np
from model import LinearRegressor

def generate_linear_data(n_samples: int, input_dim: int, noise_std: float = 0.1):
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples (int): Number of data points
        input_dim (int): Number of input features
        noise_std (float): Standard deviation of Gaussian noise

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features), y (targets)
    """
    np.random.seed(42)

    # Generate random features
    X = np.random.randn(n_samples, input_dim)

    # Generate true weights and bias
    true_w = np.random.randn(input_dim, 1)
    true_b = np.random.randn()

    # Generate noise
    noise = noise_std * np.random.randn(n_samples, 1)

    # Compute targets
    y = X @ true_w + true_b + noise

    return X, y

def train(model: LinearRegressor, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict(X)

        # Compute loss
        loss = model.compute_loss(X, y)

        # Compute gradients
        dw, db = model.compute_gradients(X, y)

        # Update parameters
        model.update_params(dw, db, lr)

        # Optional logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

if __name__ == "__main__":
    input_dim = 5
    n_samples = 200

    # Generate data
    X, y = generate_linear_data(n_samples, input_dim)

    # Initialize model (choose 'l1' or 'l2')
    model = LinearRegressor(input_dim=input_dim, regularization='l2', lambda_=0.1)

    # Train model
    train(model, X, y, epochs=100, lr=0.01)

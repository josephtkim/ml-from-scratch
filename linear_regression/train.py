import numpy as np
from model import LinearRegression

def generate_dummy_data(n_samples: int, input_dim: int):
    """
    Generates synthetic data for linear regression.
    Returns:
        X: (n_samples, input_dim)
        y: (n_samples, 1)
        true_w: (input_dim, 1)
        true_b: float
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    true_w = np.random.randn(input_dim, 1)
    true_b = 2.0
    noise = 0.1 * np.random.randn(n_samples, 1)

    y = X @ true_w + true_b + noise
    return X, y, true_w, true_b

def train(model: LinearRegression, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict(X)

        # Loss computation
        loss = model.compute_loss(y_pred, y)

        # Gradient computation
        grad_w, grad_b = model.compute_gradients(X, y_pred, y)

        # Parameter update
        model.update_parameters(grad_w, grad_b)

        # Optional: print or log loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

if __name__ == "__main__":
    input_dim = 3
    n_samples = 100

    # Prepare data
    X, y, true_w, true_b = generate_dummy_data(n_samples, input_dim)

    # Initialize and train model
    model = LinearRegression(input_dim=input_dim, learning_rate=0.01)
    train(model, X, y, epochs=300)

    # Evaluation
    print("\n=== Final Evaluation ===")
    y_pred = model.predict(X)
    final_loss = model.compute_loss(y_pred, y)
    print(f"Final Loss: {final_loss:.4f}\n")

    # Compare learned vs. true parameters
    print("True Weights:\n", true_w.flatten())
    print("Learned Weights:\n", model.weights.flatten())
    print("\nTrue Bias:", true_b)
    print("Learned Bias:", model.bias)
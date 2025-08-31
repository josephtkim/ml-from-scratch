import numpy as np
from model import LinearRegression

def generate_dummy_data(n_samples: int, input_dim: int):
    """
    Generates synthetic data for linear regression.
    Returns:
        X: (n_samples, input_dim)
        y: (n_samples, 1)
    """
    # TODO: Generate random data X, true weights, and y = Xw + b + noise
    pass

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
    X, y = generate_dummy_data(n_samples, input_dim)

    # Initialize and train model
    model = LinearRegression(input_dim=input_dim, learning_rate=0.01)
    train(model, X, y, epochs=100)

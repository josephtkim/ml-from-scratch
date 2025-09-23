import numpy as np
from model import Perceptron

def generate_dummy_linear_data(n_samples: int, input_dim: int):
    """
    Generate linearly separable data for binary classification.
    Returns:
        X, y, true_w, true_b
    """
    np.random.seed(42)
    
    # Generate random features
    X = np.random.randn(n_samples, input_dim)
    
    # Choose a "true" linear separator 
    true_w = np.random.randn(input_dim, 1)
    true_b = np.random.randn()
    
    # Compute linear scores 
    z = X @ true_w + true_b 
    
    # Assign labels {-1, +1} depending on sign of z 
    y = np.where(z >= 0, 1, -1).reshape(-1, 1)
    
    return X, y, true_w, true_b 

def train(model: Perceptron, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict(X)

        # Compute loss
        loss = model.compute_loss(X, y)

        # Compute gradients (or parameter deltas for misclassified samples)
        dw, db = model.compute_gradients(X, y)

        # Update model parameters
        model.update_params(dw, db)

        # Optional: Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

if __name__ == "__main__":
    input_dim = 2
    n_samples = 100

    # Generate synthetic dataset
    X, y, true_w, true_b = generate_dummy_linear_data(n_samples, input_dim)

    # Initialize model
    model = Perceptron(input_dim=input_dim, lr=1.0)

    # Train model
    train(model, X, y, epochs=100)

    # Evaluate accuracy
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"\nAccuracy: {acc:.4f}")

    print("True w:", true_w.flatten())
    print("Learned w:", model.w.flatten())
    print("True b:", true_b)
    print("Learned b:", model.b)

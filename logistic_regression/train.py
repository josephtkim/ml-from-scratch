import numpy as np
from model import LogisticRegression

def generate_dummy_classification_data(n_samples: int, input_dim: int):
    """
    Generates synthetic data for binary classification.
    Returns:
        X: (n_samples, input_dim)
        y: (n_samples, 1) - binary labels {0, 1}
        true_w: (input_dim, 1)
        true_b: float
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, input_dim)
    true_w = np.random.randn(input_dim, 1)
    true_b = -0.5

    # Linear output + noise
    logits = X @ true_w + true_b + 0.1 * np.random.randn(n_samples, 1)

    # Apply sigmoid and threshold at 0.5
    probs = 1 / (1 + np.exp(-logits))
    y = (probs > 0.5).astype(int)

    return X, y, true_w, true_b

def train(model: LogisticRegression, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict_proba(X)

        # Loss computation
        loss = model.compute_loss(X, y)

        # Gradient computation
        grad_w, grad_b = model.compute_gradients(X, y)

        # Parameter update
        model.update_params(grad_w, grad_b)

        # Optional: print or log loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

if __name__ == "__main__":
    input_dim = 3
    n_samples = 100

    # Prepare data
    X, y, true_w, true_b = generate_dummy_classification_data(n_samples, input_dim)

    # Initialize and train model
    model = LogisticRegression(input_dim=input_dim, lr=0.01)
    train(model, X, y, epochs=3000)

    # Evaluation
    print("\n=== Final Evaluation ===")
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Compare learned vs. true parameters
    print("True Weights:\n", true_w.flatten())
    print("Learned Weights:\n", model.w.flatten())
    print("\nTrue Bias:", true_b)
    print("Learned Bias:", model.b)

import numpy as np
from model import MLPClassifier

def generate_dummy_data(n_samples, input_dim, output_dim, seed: int = 42):
    """
    Generate two Gaussian blobs centered apart for binary classification.
    """
    np.random.seed(seed)
    assert output_dim == 2, "This version is for binary classification only."

    # Split evenly between classes
    half = n_samples // 2

    # Class 0: centered at -1
    X0 = np.random.randn(half, input_dim) + (-1.5)
    y0 = np.zeros((half,), dtype=int)

    # Class 1: centered at +1
    X1 = np.random.randn(n_samples - half, input_dim) + (+1.5)
    y1 = np.ones((n_samples - half,), dtype=int)

    X = np.vstack([X0, X1])
    y_idx = np.concatenate([y0, y1])

    # One-hot encode
    y = np.zeros((n_samples, output_dim))
    y[np.arange(n_samples), y_idx] = 1

    return X, y

def train(model: MLPClassifier, X: np.ndarray, y: np.ndarray, epochs: int = 100):
    model.fit(X, y)

if __name__ == "__main__":
    n_samples = 200

    # Model dimensions
    input_dim = 2
    hidden_dim = 16
    output_dim = 2
    layer_sizes = [input_dim, hidden_dim, output_dim]

    # Generate data
    X, y = generate_dummy_data(n_samples=n_samples, input_dim=input_dim, output_dim=output_dim)

    # Initialize model with Xavier or He initialization
    model = MLPClassifier(layer_sizes=layer_sizes, learning_rate=0.01, epochs=100, init_type="xavier", seed=42)

    # Train model (delegates to model.fit)
    train(model, X, y, epochs=model.epochs)

    # Simple evaluation
    y_pred_idx = model.predict(X)
    y_true_idx = np.argmax(y, axis=1)
    acc = np.mean(y_pred_idx == y_true_idx)
    print(f"\nTraining accuracy: {acc:.4f}")

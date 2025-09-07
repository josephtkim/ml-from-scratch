import numpy as np
from model import DecisionTreeClassifier

def generate_dummy_classification_data(n_samples: int = 100, n_features: int = 2, seed: int = 42):
    """
    Generates synthetic linearly separable classification data (binary classes).
    Args:
        n_samples: number of total samples
        n_features: number of input features
        seed: random seed for reproducibility
    Returns:
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with binary labels (0 or 1)
    """
    np.random.seed(seed)
    half = n_samples // 2
    
    # Class 0: centered at (-1, -1, ..., -1)
    X0 = np.random.randn(half, n_features) + (-2.0)
    y0 = np.zeros(half, dtype=int)
    
    # Class 1: centered at (+1, +1, ..., +1)
    X1 = np.random.randn(n_samples - half, n_features) + 2.0 
    y1 = np.ones(n_samples - half, dtype=int)
    
    # Combine and shuffle
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

def train(model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray):
    """
    Trains the model.
    """
    model.fit(X, y)

def evaluate(model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray):
    """
    Evaluates accuracy of the model.
    """
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    # Parameters
    n_samples = 200
    n_features = 2

    # Data generation
    X, y = generate_dummy_classification_data(n_samples, n_features)

    # Initialize model
    tree = DecisionTreeClassifier(max_depth=5)

    # Train
    train(tree, X, y)

    # Evaluate
    evaluate(tree, X, y)

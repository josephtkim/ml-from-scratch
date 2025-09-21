import numpy as np
from model import MLPClassifier

def generate_dummy_data(n_samples: int = 200, n_features: int = 2, seed: int = 0):
    """
    Generate synthetic binary classification data.
    """
    np.random.seed(seed)
    X0 = np.random.randn(n_samples // 2, n_features) + (-2)
    y0 = np.zeros((n_samples // 2,))

    X1 = np.random.randn(n_samples // 2, n_features) + 2
    y1 = np.ones((n_samples // 2,))

    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer labels (shape: [n_samples]) to one-hot encoded matrix (shape: [n_samples, num_classes]).
    """
    y = y.astype(int)  # Ensure integer indexing
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def evaluate(model: MLPClassifier, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    X, y = generate_dummy_data()
    y_onehot = one_hot_encode(y, num_classes=2)

    mlp = MLPClassifier(layer_sizes=[2, 16, 2], learning_rate=0.01, epochs=100)
    mlp.fit(X, y_onehot)
    evaluate(mlp, X, y)

import numpy as np
from model import RandomForestClassifier

def generate_dummy_classification_data(n_samples: int = 100, n_features: int = 2, seed: int = 42):
    """
    Creates simple synthetic binary classification dataset.
    """
    np.random.seed(seed)
    half = n_samples // 2
    X0 = np.random.randn(half, n_features) + (-2)
    y0 = np.zeros(half)
    X1 = np.random.randn(n_samples - half, n_features) + 2
    y1 = np.ones(n_samples - half)
    X = np.vstack([X0, X1])
    y = np.concatenate([y0, y1])
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

def train(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray):
    model.fit(X, y)

def evaluate(model: RandomForestClassifier, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print(f"Accuracy: {acc:.2f}")

if __name__ == "__main__":
    X, y = generate_dummy_classification_data(n_samples=200, n_features=2)
    model = RandomForestClassifier(n_estimators=10, max_depth=5)
    train(model, X, y)
    evaluate(model, X, y)

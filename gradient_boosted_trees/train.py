import numpy as np
from model import GradientBoostedTrees

def generate_dummy_data(n_samples: int = 200, n_features: int = 2, seed: int = 42):
    """
    Generates simple binary classification data (y ∈ {0, 1}).
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

def train(model: GradientBoostedTrees, X: np.ndarray, y: np.ndarray):
    model.fit(X, y)

def evaluate(model: GradientBoostedTrees, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)

    if model.loss_name == "log_loss":
        acc = np.mean(y_pred == y)
        print(f"Accuracy: {acc:.2f}")
    elif model.loss_name == "mse":
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        print(f"RMSE: {rmse:.2f}")
    else:
        raise ValueError("Unsupported loss function for evaluation.")

if __name__ == "__main__":
    X, y = generate_dummy_data(n_samples=200, n_features=2)

    model = GradientBoostedTrees(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="log_loss"  # or "mse"
    )

    train(model, X, y)
    evaluate(model, X, y)

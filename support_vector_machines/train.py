import numpy as np
import matplotlib.pyplot as plt
from model import SupportVectorMachine

def generate_dummy_data(n_samples=100, n_features=2, seed=42):
    """
    Generates linearly separable 2-class data.
    Returns:
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with binary labels {0, 1}
    """
    np.random.seed(seed)
    X_pos = np.random.randn(n_samples // 2, n_features) + 2
    y_pos = np.ones(n_samples // 2)
    X_neg = np.random.randn(n_samples // 2, n_features) - 2
    y_neg = np.zeros(n_samples // 2)

    X = np.vstack((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]

def train(model: SupportVectorMachine, X: np.ndarray, y: np.ndarray):
    model.fit(X, y, n_iters=1000)

def evaluate(model: SupportVectorMachine, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y.reshape(-1, 1))  # shape-safe comparison
    print(f"Accuracy: {acc:.2f}")

def plot_decision_boundary(model: SupportVectorMachine, X: np.ndarray, y: np.ndarray):
    """
    Visualizes decision boundary in 2D.
    Only works if input_dim = 2.
    """
    if X.shape[1] != 2:
        print("Skipping plot (not 2D data).")
        return

    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("SVM Decision Boundary")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.show()

if __name__ == "__main__":
    X, y = generate_dummy_data(n_samples=200, n_features=2)
    model = SupportVectorMachine(input_dim=2, lr=0.01, C=1.0)
    train(model, X, y)
    evaluate(model, X, y)
    plot_decision_boundary(model, X, y)

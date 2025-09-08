import numpy as np

class SupportVectorMachine:
    def __init__(self, input_dim: int, lr: float = 0.01, C: float = 1.0):
        self.weights = np.zeros((input_dim, 1))
        self.bias = 0.0
        self.lr = lr
        self.C = C  # Regularization strength

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts binary class labels using sign(w^T x + b).
        Args:
            X: shape (batch_size, input_dim)
        Returns:
            predictions: shape (batch_size, 1), values in {0, 1}
        """
        raw = X @ self.weights + self.bias 
        return (raw >= 0).astype(int)

    def compute_loss(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes hinge loss with L2 regularization.
        Args:
            X: shape (batch_size, input_dim)
            y_true: shape (batch_size, 1), values in {0, 1}
        Returns:
            loss: scalar
        """
        y = np.where(y_true <= 0, -1, 1) # Convert to (-1, +1)
        scores = X @ self.weights + self.bias 
        margins = 1 - y * scores 
        hinge_loss = np.maximum(0, margins)
        loss = 0.5 * np.sum(self.weights ** 2) + self.C * np.mean(hinge_loss)
        return loss

    def compute_gradients(self, X: np.ndarray, y_true: np.ndarray):
        """
        Computes gradients of hinge loss + L2 regularization.
        Args:
            X: shape (batch_size, input_dim)
            y_true: shape (batch_size, 1), values in {0, 1}
        Returns:
            grad_w: shape (input_dim, 1)
            grad_b: scalar
        """
        y = np.where(y_true <= 0, -1, 1)
        scores = X @ self.weights + self.bias 
        margins = 1 - y * scores 
        mask = (margins > 0).astype(float) # 1 where margin violated 
        
        # Broadcast-safe:
        grad_w = self.weights - self.C * (X.T @ (mask * y)) / X.shape[0]
        grad_b = -self.C * np.mean(mask * y)
        
        return grad_w, grad_b

    def update_parameters(self, grad_w: np.ndarray, grad_b: float):
        """
        Updates model parameters using gradient descent.
        """
        self.weights -= self.lr * grad_w
        self.bias -= self.lr * grad_b

    def fit(self, X: np.ndarray, y: np.ndarray, n_iters: int = 1000):
        """
        Trains the SVM using subgradient descent.
        Args:
            X: input features, shape (n_samples, input_dim)
            y: labels, shape (n_samples,) or (n_samples, 1), values in {0, 1}
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for _ in range(n_iters):
            y_pred = X @ self.weights + self.bias 
            loss = self.compute_loss(X, y)
            grad_w, grad_b = self.compute_gradients(X, y)
            self.update_parameters(grad_w, grad_b)
            
            if _ % 100 == 0:
                print(f"Iter {_}: loss = {loss:.4f}")
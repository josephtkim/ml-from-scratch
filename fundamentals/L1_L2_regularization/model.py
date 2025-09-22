import numpy as np

class LinearRegressor:
    def __init__(self, input_dim: int, regularization: str = 'l2', lambda_: float = 0.1):
        """
        Initialize weights and config.
        """
        self.w = np.zeros((input_dim, 1))  # shape: (d, 1)
        self.b = 0.0
        self.regularization = regularization  # 'l1' or 'l2'
        self.lambda_ = lambda_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions.
        """
        z = X @ self.w + self.b
        return z 

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute regularized loss (MSE + L1/L2).
        """
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        
        mse_loss = (1 / n_samples) * np.sum((y - y_pred) ** 2)
        
        if self.regularization == 'l1':
            reg = self.lambda_ * np.sum(np.abs(self.w))
        elif self.regularization == 'l2':
            reg = self.lambda_ * np.sum(self.w ** 2)
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization}")
        
        return mse_loss + reg

    def compute_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        Compute gradients of loss w.r.t. weights and bias.
        
        Args:
            X (np.ndarray): Input data, shape (n_samples, input_dim)
            y (np.ndarray): Target values, shape (n_samples, 1)
        
        Returns:
            Tuple[np.ndarray, float]: Gradients for w and b
        """
        n_samples = X.shape[0]
        
        # Compute predictions
        y_pred = X @ self.w + self.b
        
        # Compute residual
        error = y - y_pred  # shape: (n_samples, 1)

        # Gradient of MSE loss
        dw = -(2 / n_samples) * X.T @ error  # shape: (input_dim, 1)
        db = -(2 / n_samples) * np.sum(error)  # scalar

        # Add regularization gradient
        if self.regularization == 'l2':
            dw += 2 * self.lambda_ * self.w
        elif self.regularization == 'l1':
            dw += self.lambda_ * np.sign(self.w)
        else:
            raise ValueError(f"Unknown regularization type: {self.regularization}")

        return dw, db

    def update_params(self, dw: np.ndarray, db: float, lr: float):
        """
        Update weights and bias using gradient descent.
        """
        self.w -= lr * dw 
        self.b -= lr * db 
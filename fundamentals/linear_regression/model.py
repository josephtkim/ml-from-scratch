import numpy as np

class LinearRegression:
    def __init__(self, input_dim: int, learning_rate: float = 0.01):
        self.weights = np.zeros((input_dim, 1))
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values given input features.
        Args:
            X: shape (batch_size, input_dim)
        Returns:
            predictions: shape (batch_size, 1)
        """
        y = X @ self.weights + self.bias
        return y

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the mean squared error loss.
        Args:
            y_pred: predicted outputs (batch_size, 1)
            y_true: ground truth outputs (batch_size, 1)
        Returns:
            loss: scalar
        """
        N = len(y_pred)
        mse = (1.0 / N) * np.sum((y_pred - y_true)**2)
        return mse 

    def compute_gradients(self, X: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Computes gradients for weights and bias.
        Args:
            X: input features (batch_size, input_dim)
            y_pred: predicted outputs (batch_size, 1)
            y_true: ground truth outputs (batch_size, 1)
        Returns:
            grad_w: shape (input_dim, 1)
            grad_b: scalar
        """
        N = X.shape[0]
        error = y_pred - y_true
        grad_w = (2.0 / N) * X.T @ error
        grad_b = (2.0 / N) * np.sum(error)
        return grad_w, grad_b 

    def update_parameters(self, grad_w: np.ndarray, grad_b: float):
        """
        Updates model parameters using gradient descent.
        Args:
            grad_w: gradient w.r.t. weights
            grad_b: gradient w.r.t. bias
        """
        self.weights -= self.lr*grad_w 
        self.bias -= self.lr*grad_b

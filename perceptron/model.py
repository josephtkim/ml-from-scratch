import numpy as np

class Perceptron:
    def __init__(self, input_dim, lr=1.0):
        """
        Initialize weights and bias.
        Args:
            input_dim (int): Number of input features.
            lr (float): Learning rate.
        """
        self.w = np.zeros((input_dim, 1))  # shape: (d, 1)
        self.b = 0.0
        self.lr = lr

    def predict(self, X):
        """
        Predict binary labels.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim).
        Returns:
            np.ndarray: Predictions (-1 or 1).
        """
        z = X @ self.w + self.b
        return np.where(z >= 0, 1, -1)

    def compute_loss(self, X, y):
        """
        Compute perceptron loss over dataset.
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
        Returns:
            float: Loss value.
        """
        z = X @ self.w + self.b
        margins = y * z
        losses = np.maximum(0, -margins)
        return np.sum(losses)

    def compute_gradients(self, X, y):
        """
        Compute weight and bias updates for misclassified samples.
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
        Returns:
            dw, db: Update directions.
        """
        n_samples = X.shape[0]
        dw = np.zeros_like(self.w)
        db = 0.0

        for i in range(n_samples):
            xi = X[i].reshape(-1, 1)  # shape: (d, 1)
            yi = y[i]

            z = float(xi.T @ self.w + self.b)

            if yi * z <= 0:  # misclassified
                dw += yi * xi
                db += yi

        return dw, db

    def update_params(self, dw, db):
        """
        Update weights and bias.
        Args:
            dw (np.ndarray): Gradient w.r.t weights.
            db (float): Gradient w.r.t bias.
        """
        self.w += self.lr * dw 
        self.b += self.lr * db 

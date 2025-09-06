import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, lr=0.01):
        """
        Initialize weights and bias.
        Args:
            input_dim (int): Number of input features.
            lr (float): Learning rate.
        """
        self.w = np.zeros((input_dim, 1))  # weights
        self.b = 0.0  # bias
        self.lr = lr

    def sigmoid(self, z):
        """
        Compute the sigmoid activation.
        Args:
            z (np.ndarray): Linear output.
        Returns:
            np.ndarray: Sigmoid activation.
        """
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        """
        Compute predicted probabilities.
        Args:
            X (np.ndarray): Input data of shape (n_samples, input_dim).
        Returns:
            np.ndarray: Probabilities for class 1.
        """
        z = X@self.w + self.b 
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels (0 or 1).
        Args:
            X (np.ndarray): Input data.
            threshold (float): Threshold for classification.
        Returns:
            np.ndarray: Binary predictions.
        """
        preds = self.predict_proba(X)
        return preds > threshold

    def compute_loss(self, X, y):
        """
        Compute binary cross-entropy loss.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True labels (0 or 1).
        Returns:
            float: Loss value.
        """
        n = X.shape[0]
        y_pred = self.predict_proba(X)
        # To avoid log(0) instability:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = - (1.0 / n) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def compute_gradients(self, X, y):
        """
        Compute gradients of weights and bias.
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
        Returns:
            dw (np.ndarray): Gradient w.r.t. weights.
            db (float): Gradient w.r.t. bias.
        """
        n = X.shape[0]
        y_pred = self.predict_proba(X)
        error = y_pred - y
        dw = (1.0 / n) * X.T @ error
        db = (1.0 / n) * np.sum(error)
        return dw, db 

    def update_params(self, dw, db):
        """
        Update weights and bias using gradient descent.
        Args:
            dw (np.ndarray): Gradient w.r.t. weights.
            db (float): Gradient w.r.t. bias.
        """
        self.w -= self.lr*dw
        self.b -= self.lr*db

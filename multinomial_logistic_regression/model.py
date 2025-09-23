import numpy as np

class MultinomialLogisticRegression:
    def __init__(self, input_dim, num_classes, lr=0.01):
        """
        Initialize weight matrix and bias vector.
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            lr (float): Learning rate.
        """
        self.W = np.zeros((num_classes, input_dim))  # shape: (K, d)
        self.b = np.zeros((num_classes, 1))          # shape: (K, 1)
        self.lr = lr

    def softmax(self, Z):
        """
        Compute the softmax probabilities.
        Args:
            Z (np.ndarray): Raw scores, shape (n_samples, K)
        Returns:
            np.ndarray: Probabilities, shape (n_samples, K)
        """
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def predict_proba(self, X):
        """
        Compute predicted class probabilities.
        Args:
            X (np.ndarray): Input data, shape (n_samples, input_dim)
        Returns:
            np.ndarray: Class probabilities, shape (n_samples, K)
        """
        z = X@self.W.T + self.b.T
        return self.softmax(z)

    def predict(self, X):
        """
        Predict class labels.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Predicted class indices, shape (n_samples,)
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def compute_loss(self, X, y_onehot):
        """
        Compute cross-entropy loss.
        Args:
            X (np.ndarray): Input features.
            y_onehot (np.ndarray): One-hot encoded true labels, shape (n_samples, K)
        Returns:
            float: Loss value.
        """
        eps = 1e-15
        y_pred = self.predict_proba(X)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss_matrix = y_onehot * np.log(y_pred)
        loss = -np.mean(np.sum(loss_matrix, axis=1))
        return loss

    def compute_gradients(self, X, y_onehot):
        """
        Compute gradients of weights and biases.
        Args:
            X (np.ndarray): Input data.
            y_onehot (np.ndarray): One-hot encoded true labels.
        Returns:
            dW (np.ndarray): Gradient w.r.t. weights, shape (K, d)
            db (np.ndarray): Gradient w.r.t. bias, shape (K, 1)
        """
        n = X.shape[0]
        y_pred = self.predict_proba(X)
        error = y_pred - y_onehot
        dW = (1/n) * error.T @ X
        db = (1/n) * np.sum(error, axis=0, keepdims=True).T # shape (K, 1)
        return dW, db

    def update_params(self, dW, db):
        """
        Update model parameters using gradient descent.
        Args:
            dW (np.ndarray): Gradient w.r.t. weights.
            db (np.ndarray): Gradient w.r.t. biases.
        """
        self.W -= self.lr*dW 
        self.b -= self.lr*db

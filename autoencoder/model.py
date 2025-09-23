import numpy as np

class Autoencoder:
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize weights for encoder and decoder.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder parameters 
        self.W_enc = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b_enc = np.zeros((1, hidden_dim))
        
        # Decoder parameters 
        self.W_dec = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_dec = np.zeros((1, input_dim))
        
        self.Z_pre = None  # Pre-activation (Z = XW + b)
        self.Z = None      # Post-activation (ReLU(Z_pre))
        self.X_hat = None

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(x.dtype)

    def encode(self, X):
        """
        Encode input into latent representation using ReLU.
        """
        self.Z_pre = X @ self.W_enc + self.b_enc
        self.Z = self.relu(self.Z_pre)
        return self.Z

    def decode(self, Z):
        """
        Decode latent representation back to input space.
        """
        self.X_hat = Z @ self.W_dec + self.b_dec
        return self.X_hat

    def compute_loss(self, X, X_hat, reduction: str = "mean"):
        """
        Compute reconstruction loss (MSE).
        """
        diff = X_hat - X
        if reduction == "mean":
            loss = np.mean(diff ** 2)
        elif reduction == "sum":
            loss = np.sum(diff ** 2)
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
        return float(loss)

    def compute_gradients(self, X, reduction: str = "mean"):
        """
        Compute gradients with ReLU in encoder.
        """
        N, D = X.shape
        Z_pre = self.Z_pre
        Z = self.Z
        X_hat = self.X_hat

        if reduction == "mean":
            s = 2.0 / (N * D)
        elif reduction == "sum":
            s = 2.0
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")
        
        # Gradients through decoder
        dXhat = s * (X_hat - X)
        dW_dec = Z.T @ dXhat
        db_dec = dXhat.sum(axis=0, keepdims=True)

        # Gradients through encoder
        dZ = dXhat @ self.W_dec.T
        dZ_relu = dZ * self.relu_derivative(Z_pre)  # Backprop through ReLU
        dW_enc = X.T @ dZ_relu
        db_enc = dZ_relu.sum(axis=0, keepdims=True)

        return {
            "dW_enc": dW_enc, "db_enc": db_enc,
            "dW_dec": dW_dec, "db_dec": db_dec
        }

    def update_params(self, grads, lr: float):
        """
        Update weights and biases.
        """
        self.W_enc -= lr * grads["dW_enc"]
        self.b_enc -= lr * grads["db_enc"]
        self.W_dec -= lr * grads["dW_dec"]
        self.b_dec -= lr * grads["db_dec"]

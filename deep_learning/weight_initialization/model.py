import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def relu_derivative(Z):
    return (Z > 0).astype(float)

class MLPClassifier:
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01, epochs: int = 100, init_type: str = "xavier", seed: int = 42):
        """
        Args:
            layer_sizes: List like [input_dim, hidden1, ..., output_dim]
            init_type: "xavier" or "he"
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.init_type = init_type
        self.seed = seed

        self.weights = {}  # Dict[layer_idx] = weight matrix (shape: [out_dim, in_dim])
        self.biases = {}   # Dict[layer_idx] = bias vector   (shape: [out_dim, 1])

    def _initialize_parameters(self):
        """
        Initialize weights and biases using Xavier or He initialization.
        """
        np.random.seed(self.seed)

        for l in range(1, len(self.layer_sizes)):
            in_dim = self.layer_sizes[l - 1]
            out_dim = self.layer_sizes[l]
            
            if self.init_type == "xavier":
                # Xavier: var = 1 / (in_dim + out_dim)
                var = 1 / (in_dim + out_dim)
            elif self.init_type == "he":
                # He/Kaiming: var = 2 / in_dim
                var = 2 / in_dim
            else:
                raise ValueError(f"Unknown init_type: {self.init_type}. Use 'xavier' or 'he'.")
                
            std = np.sqrt(var)
            
            # Initialize parameters 
            self.weights[l] = np.random.randn(out_dim, in_dim) * std 
            self.biases[l] = np.zeros((out_dim, 1))
           
    def _forward(self, X: np.ndarray) -> tuple[dict, dict]:
        """
        Forward pass through all layers.
        Returns:
            activations: {0: input, 1: A1, ..., L: output}
            pre_activations: {1: Z1, ..., L: ZL}
        """
        activations = {}
        pre_activations = {}
        A = X 
        activations[0] = A
        
        for l in range(1, len(self.layer_sizes)):
            Z = A @ self.weights[l].T + self.biases[l].T 
            pre_activations[l] = Z 
            
            if l == len(self.layer_sizes) - 1:
                A = softmax(Z)
            else:
                A = relu(Z)
                
            activations[l] = A 

        return activations, pre_activations

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss for classification.
        """
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def _backward(self, activations: dict, pre_activations: dict, y_true: np.ndarray) -> tuple[dict, dict]:
        """
        Backward pass to compute gradients.
        Returns:
            grad_w: dict of gradients for weights
            grad_b: dict of gradients for biases
        """
        grad_w = {}
        grad_b = {}
        L = len(self.layer_sizes) - 1
        m = y_true.shape[0]
        
        delta = activations[L] - y_true # softmax + CE 
        
        for l in reversed(range(1, L + 1)):
            A_prev = activations[l - 1]
            grad_w[l] = (delta.T @ A_prev) / m
            grad_b[l] = np.mean(delta, axis=0, keepdims=True).T 
            
            if l > 1:
                Z_prev = pre_activations[l - 1]
                delta = (delta @ self.weights[l]) * relu_derivative(Z_prev)
                
        return grad_w, grad_b 

    def _update_parameters(self, grad_w: dict, grad_b: dict):
        """
        Update weights and biases with gradient descent.
        """
        for l in self.weights:
            self.weights[l] -= self.learning_rate * grad_w[l]
            self.biases[l] -= self.learning_rate * grad_b[l]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train MLP using forward-backward passes and parameter updates.
        """
        self._initialize_parameters()

        for epoch in range(self.epochs):
            activations, pre_activations = self._forward(X)
            loss = self._compute_loss(y, activations[len(self.layer_sizes) - 1])
            grad_w, grad_b = self._backward(activations, pre_activations, y)
            self._update_parameters(grad_w, grad_b)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the trained model.
        """
        activations, _ = self._forward(X)
        output = activations[len(self.layer_sizes) - 1]
        return np.argmax(output, axis=1)

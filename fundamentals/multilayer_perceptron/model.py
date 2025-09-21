import numpy as np

def relu(Z):
    return np.maximum(0, Z)
    
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)
    
def relu_derivative(Z):
    return (Z > 0).astype(float)

class MLPClassifier:
    def __init__(self, layer_sizes: list[int], learning_rate: float = 0.01, epochs: int = 100, seed: int = 42):
        """
        layer_sizes: [input_dim, hidden1_dim, ..., output_dim]
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.weights = {}  # Dict[layer_idx] = weight_matrix
        self.biases = {}   # Dict[layer_idx] = bias_vector

    def _initialize_parameters(self):
        """
        Initialize weights and biases using small random values.
        """
        np.random.seed(self.seed)
        
        for l in range(1, len(self.layer_sizes)):
            input_dim = self.layer_sizes[l - 1]
            output_dim = self.layer_sizes[l]
            
            self.weights[l] = np.random.randn(output_dim, input_dim) * 0.01
            self.biases[l] = np.zeros((output_dim, 1))

    def _forward(self, X: np.ndarray) -> dict:
        """
        Perform forward pass through all layers.
        Returns dictionary of activations for each layer.
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
        Compute average cross-entropy loss over all samples.
        
        y_true: one-hot encoded labels, shape (n_samples, n_classes)
        y_pred: softmax outputs, shape (n_samples, n_classes)
        """
        # Avoid log(0) by adding epsilon for numerical stability
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Cross-entropy: -sum(y_true * log(y_pred)) averaged over all samples
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def _backward(self, activations: dict, pre_activations: dict, y: np.ndarray) -> tuple[dict, dict]:
        """
        Perform backpropagation to compute gradients of loss w.r.t. weights and biases.
        
        activations: forward-pass outputs {0: X, 1: A1, ..., L: output probs}
        y: true one-hot encoded labels
        """
        grad_w = {}
        grad_b = {}
        L = len(self.layer_sizes) - 1
        m = y.shape[0]
        
        delta = activations[L] - y  # softmax + CE

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
        Apply gradient descent update step using computed gradients.
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using forward pass.
        """
        activations, _ = self._forward(X)  # Only need activations
        output = activations[len(self.layer_sizes) - 1]
        return np.argmax(output, axis=1)

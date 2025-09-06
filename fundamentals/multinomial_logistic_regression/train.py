import numpy as np
from model import MultinomialLogisticRegression

def generate_dummy_multiclass_data(n_samples: int, input_dim: int, num_classes: int):
    """
    Generate synthetic data for multiclass classification with learnable structure.
    Returns:
        X: (actual_n_samples, input_dim)
        y: (actual_n_samples,) - integer class labels in {0, ..., K-1}
        y_onehot: (actual_n_samples, num_classes)
        true_W: (num_classes, input_dim)
        true_b: (num_classes, 1)
    """
    np.random.seed(42)

    samples_per_class = n_samples // num_classes
    actual_n_samples = samples_per_class * num_classes  # to ensure equal class counts

    X = []
    y = []

    for k in range(num_classes):
        center = np.random.randn(input_dim) * 3.0  # spread-out class centers
        class_samples = np.random.randn(samples_per_class, input_dim) + center
        X.append(class_samples)
        y.extend([k] * samples_per_class)

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle
    perm = np.random.permutation(actual_n_samples)
    X = X[perm]
    y = y[perm]

    # One-hot encode labels
    y_onehot = np.zeros((actual_n_samples, num_classes))
    y_onehot[np.arange(actual_n_samples), y] = 1

    # Generate true weights and biases
    true_W = np.random.randn(num_classes, input_dim)
    true_b = np.random.randn(num_classes, 1)

    return X, y, y_onehot, true_W, true_b

def train(model: MultinomialLogisticRegression, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 100):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model.predict_proba(X)

        # Loss computation
        loss = model.compute_loss(X, y_onehot)
        
        # Gradient computation 
        grad_w, grad_b = model.compute_gradients(X, y_onehot)
        
        # Parameter update
        model.update_params(grad_w, grad_b)
        
        # Print or log loss 
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
        
if __name__ == "__main__":
    input_dim = 4
    num_classes = 3
    n_samples = 200

    # Generate data
    X, y, y_onehot, true_W, true_b = generate_dummy_multiclass_data(n_samples, input_dim, num_classes)

    # Initialize model
    model = MultinomialLogisticRegression(input_dim=input_dim, num_classes=num_classes, lr=0.1)

    # Train
    train(model, X, y_onehot, epochs=3000)

    # Evaluation
    print("\n=== Final Evaluation ===")
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.4f}")

    # Compare learned vs. true parameters
    print("\n=== True vs Learned Weights ===")
    for k in range(num_classes):
        print(f"\nClass {k}:")
        print("True W:", true_W[k])
        print("Learned W:", model.W[k])
        print("True b:", true_b[k][0])
        print("Learned b:", model.b[k][0])
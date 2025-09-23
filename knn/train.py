import numpy as np
from model import KNearestNeighbors

def generate_dummy_data(n_samples=100, input_dim=2, num_classes=3, seed=42):
    """
    Generate synthetic data for multiclass classification using well-separated clusters.

    Args:
        n_samples (int): Total number of samples to generate.
        input_dim (int): Number of features.
        num_classes (int): Number of distinct classes.
        seed (int): Random seed for reproducibility.

    Returns:
        X (np.ndarray): Feature matrix, shape (n_samples, input_dim)
        y (np.ndarray): Integer labels, shape (n_samples,)
    """
    np.random.seed(seed)

    samples_per_class = n_samples // num_classes
    remainder = n_samples % num_classes
    counts = np.full(num_classes, samples_per_class)
    counts[:remainder] += 1  # distribute extras

    X_list = []
    y_list = []

    scale = 3.0
    for k in range(num_classes):
        center = np.random.randn(input_dim) * scale 
        class_samples = np.random.randn(samples_per_class, input_dim) + center 
        X_list.append(class_samples)
        y_list.extend([k] * samples_per_class)

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    return X, y

def evaluate_accuracy(y_true, y_pred):
    """
    Simple accuracy metric.
    """
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    # Parameters
    input_dim = 2
    num_classes = 3
    n_samples = 150
    k = 5
    seed = 42

    # Generate dataset
    X, y = generate_dummy_data(n_samples=n_samples, input_dim=input_dim, num_classes=num_classes, seed=seed)

    # Train/test split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X = X[idx]
    y = y[idx]
    
    n_train = int(0.8 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Initialize model
    knn = KNearestNeighbors(k=k)

    # "Train" the model (store data)
    knn.fit(X_train, y_train)

    # Predict on test set
    y_pred = knn.predict(X_test)

    # Evaluate
    acc = evaluate_accuracy(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
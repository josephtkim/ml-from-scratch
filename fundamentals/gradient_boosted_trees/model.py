import numpy as np
from typing import Callable, List

class GradientBoostedTrees:
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 loss: str = 'log_loss'):  # or 'mse'
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss_name = loss
        self.trees = []
        self.loss_fn = self._select_loss_fn()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit gradient boosted trees to the training data.
        """
        self.init_value = self.loss_fn["init"](y)
        y_pred = np.full(len(y), self.init_value)

        for i in range(self.n_estimators):
            residuals = self._compute_residuals(y, y_pred)

            tree = self._DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts final output using additive ensemble of trees.
        """
        # 1. Start with initial prediction
        y_pred = np.full(X.shape[0], self._initial_prediction_value())

        # 2. Add predictions from all trees
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        # 3. Post-process if classification
        if self.loss_name == 'log_loss':
            probs = self._sigmoid(y_pred)
            return (probs >= 0.5).astype(int)
        return y_pred

    def _initial_prediction(self, y: np.ndarray) -> np.ndarray:
        """
        Initialize predictions depending on task.
        """
        if self.loss_name == "mse":
            init_value = np.mean(y)
        elif self.loss_name == "log_loss":
            p = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
            init_value = np.log(p / (1 - p))
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
            
        return np.full(y.shape[0], init_value)

    def _initial_prediction_value(self):
        if self.loss_name == "mse":
            return self.init_value 
        elif self.loss_name == "log_loss":
            return self.init_value 
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")
        
    def _compute_residuals(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Return the negative gradient of the loss with respect to the current
        raw predictions y_pred (a.k.a. pseudo-residuals).

        For 'mse' (regression):
            L = (1/2) * (y - F)^2  =>  dL/dF = F - y  =>  -grad = y - F

        For 'log_loss' (binary classification with y in {0,1}):
            p = sigmoid(F)
            L = -[ y*log(p) + (1-y)*log(1-p) ]  =>  dL/dF = p - y  =>  -grad = y - p
        """
        if self.loss_name == "mse":
            # Residuals are simply the targets minus current predictions.
            return y - y_pred
        elif self.loss_name == "log_loss":
            # Work in raw score space: convert to probabilities with sigmoid.
            p = 1.0 / (1.0 + np.exp(-y_pred))
            # Pseudo-residuals are y - p
            return y - p
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

    def _select_loss_fn(self):
        """
        Return a dictionary of loss utilities based on self.loss_name.
        Each bundle exposes:
          - init(y): scalar initial prediction in raw score space
          - neg_grad(y, f): negative gradient wrt raw predictions f
          - link(f): post-process raw scores for final outputs
          - name: loss name (string)
        """
        eps = 1e-15 
        
        def _sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))
            
        if self.loss_name == "mse":
            return {
                "name": "mse",
                "init": lambda y: float(np.mean(y)),
                "neg_grad": lambda y, f: y - f,         # y - F
                "link": lambda f: f,                    # identity for regression
            }
        elif self.loss_name == "log_loss":
            return {
                "name": "log_loss",
                "init": lambda y: float(
                    np.log(
                        np.clip(np.mean(y), eps, 1 - eps)
                        / (1.0 - np.clip(np.mean(y), eps, 1 - eps))
                    )
                ),                                         # log-odds of prior
                "neg_grad": lambda y, f: y - _sigmoid(f),  # y - p 
                "link": _sigmoid,                          # map raw scores -> probs
            }
        else:
            raise ValueError(f"Unsupported loss: {self.loss_name}")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    class _DecisionTree:
        def __init__(self, max_depth: int, min_samples_split: int):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.root = None

        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            Builds a CART-style regression tree (for residuals).
            """
            self.root = self._build_tree(X, y, depth=0)

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predicts real-valued outputs (leaf means).
            """
            return np.array([self._traverse_tree(x, self.root) for x in X])

        # -----------------------
        # Internal: build / split
        # -----------------------
        def _build_tree(self, X, y, depth):
            # 1) Stopping conditions (mirror your original structure)
            if self._is_pure(y) or (self.max_depth is not None and depth >= self.max_depth) \
               or len(y) < self.min_samples_split:
                return {'type': 'leaf', 'value': self._leaf_value(y)}

            # 2) Find best feature and threshold by MSE reduction
            feature_idx, threshold = self._best_split(X, y)
            if feature_idx is None:
                return {'type': 'leaf', 'value': self._leaf_value(y)}

            # 3) Partition
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]

            # 4) Recurse
            left_child = self._build_tree(X_left, y_left, depth + 1)
            right_child = self._build_tree(X_right, y_right, depth + 1)

            # 5) Return decision node (same shape as your original)
            return {
                'type': 'decision',
                'feature': feature_idx,
                'threshold': threshold,
                'left': left_child,
                'right': right_child
            }

        def _best_split(self, X, y):
            """
            Choose split that maximizes parent MSE - weighted child MSE (variance reduction).
            Uses unique feature values as candidate thresholds, like your original code.
            """
            n_samples, n_features = X.shape
            if n_samples < 2:
                return None, None

            parent_mse = self._mse(y)
            best_gain = -np.inf
            best_feature = None
            best_threshold = None

            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold
                    right_mask = ~left_mask

                    if not left_mask.any() or not right_mask.any():
                        continue

                    y_left, y_right = y[left_mask], y[right_mask]

                    # Weighted child MSE
                    w_left = len(y_left) / n_samples
                    w_right = 1.0 - w_left
                    children_mse = w_left * self._mse(y_left) + w_right * self._mse(y_right)

                    gain = parent_mse - children_mse
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold

            return best_feature, best_threshold

        # -----------------------
        # Internal: utilities
        # -----------------------
        def _is_pure(self, y):
            # Treat near-constant targets as pure; mirrors your original purity check idea
            return len(np.unique(y)) == 1

        def _mse(self, y):
            if len(y) == 0:
                return 0.0
            mean = np.mean(y)
            return np.mean((y - mean) ** 2)

        def _leaf_value(self, y):
            # For boosting residuals, the optimal leaf value (LS) is the mean
            return float(np.mean(y)) if len(y) else 0.0

        def _traverse_tree(self, x, node):
            if node['type'] == 'leaf':
                return node['value']
            if x[node['feature']] <= node['threshold']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])

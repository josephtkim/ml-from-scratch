import numpy as np
from collections import Counter
from typing import List

class RandomForestClassifier:
    def __init__(self, n_estimators: int = 100, max_depth: int = None, min_samples_split: int = 2, max_features: str = 'sqrt', bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # 'sqrt', 'log2', or int
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains multiple independent decision trees on bootstrap samples and random feature subsets.
        """
        n_samples, n_features = X.shape
        self.trees = []
        
        for _ in range(self.n_estimators):
            # 1. Bootstrap sample 
            # respect bootstrap flag
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # 2. Random subset of features 
            feature_indices = self._select_feature_indices(n_features)
            
            # 3. Train internal decision tree on selected features 
            tree = self._DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split 
            )
            
            # Slice selected features only 
            X_subset = X_sample[:, feature_indices]
            tree.fit(X_subset, y_sample)
            
            # 4. Save both tree and its feature indices 
            self.trees.append((tree, feature_indices))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregates predictions from each tree using majority vote.
        Deterministic tie-break: pick the smallest label in case of a tie.
        """
        if not self.trees:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        n_samples = X.shape[0]
        all_tree_preds = []

        # Collect predictions from each tree
        for tree, feature_indices in self.trees:
            X_subset = X[:, feature_indices]
            preds = tree.predict(X_subset)  # shape: (n_samples,)
            all_tree_preds.append(preds)

        # (n_trees, n_samples)
        all_tree_preds = np.array(all_tree_preds)

        final_preds = []
        for i in range(n_samples):
            sample_preds = all_tree_preds[:, i]
            counts = Counter(sample_preds)
            max_count = max(counts.values())
            # Resolve ties deterministically by choosing the smallest class label
            winners = [cls for cls, c in counts.items() if c == max_count]
            final_preds.append(min(winners))

        return np.array(final_preds)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        Returns a randomly bootstrapped dataset.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def _select_feature_indices(self, n_features: int) -> np.ndarray:
        """
        Selects random subset of features per tree.
        """
        if isinstance(self.max_features, int):
            k = self.max_features
        elif self.max_features == 'sqrt':
            k = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            k = int(np.log2(n_features))
        else:
            k = n_features
        k = max(1, min(k, n_features))  # ensure at least 1 feature
        
        return np.random.choice(n_features, size=k, replace=False)

    class _DecisionTree:
        def __init__(self, max_depth: int, min_samples_split: int):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.root = None

        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            Builds the decision tree recursively.
            """
            self.root = self._build_tree(X, y, depth=0)

        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predicts class labels for input samples.
            """
            return np.array([self._traverse_tree(x, self.root) for x in X])

        def _build_tree(self, X, y, depth):
            # 1. Stopping conditions
            if self._is_pure(y) or (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
                return {'type': 'leaf', 'class': self._majority_class(y)}
                
            # 2. Find best feature and threshold 
            feature_idx, threshold = self._best_split(X, y)
            if feature_idx is None:
                return {'type': 'leaf', 'class': self._majority_class(y)}
                
            # 3. Partition the data 
            left_mask = X[:, feature_idx] <= threshold 
            right_mask = X[:, feature_idx] > threshold 
            X_left, y_left = X[left_mask], y[left_mask]
            X_right, y_right = X[right_mask], y[right_mask]
            
            # 4. Recursively build subtrees
            left_child = self._build_tree(X_left, y_left, depth + 1)
            right_child = self._build_tree(X_right, y_right, depth + 1)
            
            # 5. Return decision node 
            return {
                'type': 'decision',
                'feature': feature_idx,
                'threshold': threshold,
                'left': left_child,
                'right': right_child
            }
            
        def _best_split(self, X, y):
            n_samples, n_features = X.shape 
            best_gain = -np.inf 
            best_feature = None
            best_threshold = None 
            parent_impurity = self._gini(y)
            
            for feature_idx in range(n_features):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    left_mask = X[:, feature_idx] <= threshold 
                    right_mask = X[:, feature_idx] > threshold 
                    y_left, y_right = y[left_mask], y[right_mask]
                    
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue 
                        
                    # Compute info gain 
                    gain = self._information_gain(y, y_left, y_right, parent_impurity)
                    
                    if gain > best_gain:
                        best_gain = gain 
                        best_feature = feature_idx 
                        best_threshold = threshold 
                        
            return best_feature, best_threshold 
            
        def _is_pure(self, y):
            return len(np.unique(y)) == 1

        def _gini(self, y):
            if len(y) == 0:
                return 0.0
            classes, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            return 1.0 - np.sum(probs ** 2)

        def _majority_class(self, y):
            counter = Counter(y)
            most_common_class, _ = counter.most_common(1)[0]
            return most_common_class
            
        def _information_gain(self, y, y_left, y_right, parent_impurity):
            p = len(y_left) / len(y)
            return parent_impurity - p * self._gini(y_left) - (1 - p) * self._gini(y_right)
            
        def _traverse_tree(self, x, node):
            if node['type'] == 'leaf':
                return node['class']
            if x[node['feature']] <= node['threshold']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])
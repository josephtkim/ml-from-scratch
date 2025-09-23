from collections import Counter
import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None  # Tree structure to be built

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the decision tree on the given dataset.
        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,)
        """
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for input samples.
        Args:
            X: shape (n_samples, n_features)
        Returns:
            preds: shape (n_samples,)
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
        
    def _traverse_tree(self, x: np.ndarray, node: dict):
        """
        Recursively traverses the tree for a single input sample.
        Args:
            x: shape (n_features,) - a single input 
            node: current node in the tree (dict)
        Returns:
            predicted class label
        """
        if node['type'] == 'leaf':
            return node['class']
            
        # Decision node: go left or right depending on threshold 
        feature = node['feature']
        threshold = node['threshold']
        
        if x[feature] <= threshold:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Recursively builds the decision tree.
        Returns:
            node: dict with 'feature', 'threshold', 'left', 'right' or 'leaf'
        """
        # 1. Check stopping criteria (pure, depth, min samples)
        if self._is_pure(y) or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'type': 'leaf', 'class': self._majority_class(y)}
                    
        # 2. Find best split 
        best_feature, best_threshold = self._best_split(X, y)
        
        # If no good split is found (e.g., feature has same values), return a leaf.
        if best_feature is None:
            return {'type': 'leaf', 'class': self._majority_class(y)}

        # 3. Partition the data
        left_indices = X[:, best_feature] <= best_threshold 
        right_indices = X[:, best_feature] > best_threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]
        
        # 4. Recursively build the left and right subtrees 
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        # 5. Return a decision node 
        return {
            'type': 'decision',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }
        
    def _is_pure(self, y: np.ndarray) -> bool:
        """
        Checks if all labels in y are the same (pure node).
        Args:
            y: shape (n_samples,)
        Returns:
            True if pure, else False
        """
        return len(np.unique(y)) == 1
        
    def _majority_class(self, y: np.ndarray):
        """
        Returns the most common class in y.
        Args:
            y: shape (n_samples,)
        Returns:
            majority class label
        """
        counter = Counter(y)
        most_common_class, _ = counter.most_common(1)[0]
        return most_common_class 
        
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Finds the best feature and threshold to split on.
        Returns:
            best_feature, best_threshold
        """
        n_samples, n_features = X.shape 
        best_gain = -np.inf 
        best_feature = None
        best_threshold = None 
        
        # Choose impurity function
        impurity_fn = self._gini # or self._entropy
        parent_impurity = impurity_fn(y)
        
        for feature_idx in range(n_features):
            # Get sorted (feature, label) pairs 
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                # Split the data 
                left_mask = X[:, feature_idx] <= threshold 
                right_mask = X[:, feature_idx] > threshold 
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Skip invalid splits 
                if len(y_left) == 0 or len(y_right) == 0:
                    continue 
                    
                # Compute information gain 
                gain = self._information_gain(y, y_left, y_right, criterion='gini')
                
                # Update best split if gain is better 
                if gain > best_gain:
                    best_gain = gain 
                    best_feature = feature_idx 
                    best_threshold = threshold 
                    
        return best_feature, best_threshold

    def _gini(self, y: np.ndarray) -> float:
        """
        Computes Gini impurity of a label distribution.
        """
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y: np.ndarray) -> float:
        """
        Computes entropy of a label distribution.
        """
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray, criterion: str = 'gini') -> float:
        """
        Computes the information gain from a potential split.
        Args:
            y: parent labels
            y_left: left child labels
            y_right: right child labels
            criterion: 'gini' or 'entropy'
        Returns:
            Information gain (float)
        """
        if criterion == 'gini':
            impurity = self._gini 
        elif criterion == 'entropy':
            impurity = self._entropy 
        else:
            raise ValueError("Unsupported criterion")
            
        p = len(y_left) / len(y)
        
        # Weighted average impurity of children is subtracted from parent impurity
        return impurity(y) - p * impurity(y_left) - (1 - p) * impurity(y_right)
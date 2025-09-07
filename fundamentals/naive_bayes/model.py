import numpy as np
from typing import List, Dict
from collections import defaultdict, Counter

class NaiveBayesTextClassifier:
    def __init__(self, smoothing: float = 1.0):
        """
        Initializes the Naive Bayes classifier.
        Args:
            smoothing: Laplace smoothing parameter
        """
        self.smoothing = smoothing
        self.class_priors = {}        # P(y)
        self.word_likelihoods = {}    # P(w|y)
        self.vocab = set()
        self.classes = set()

    def fit(self, X: List[List[str]], y: List[str]):
        """
        Trains the model using the training data.
        Args:
            X: List of tokenized documents (list of word lists)
            y: List of class labels
        """
        self.classes = set(y)
        total_docs = len(y)
        class_counts = Counter(y)
        
        # Initialize data structures 
        word_counts_per_class = defaultdict(lambda: defaultdict(int)) # class -> word -> count
        total_words_per_class = defaultdict(int) # class -> total word count
        
        for doc, label in zip(X, y):
            for word in doc:
                word_counts_per_class[label][word] += 1
                total_words_per_class[label] += 1
                self.vocab.add(word)
                
        vocab_size = len(self.vocab)
        
        # Compute class priors: P(y)
        self.class_priors = {
            cls: count / total_docs 
            for cls, count in class_counts.items()
        }
        
        # Compute word likelihoods with Laplace smoothing: P(w|y)
        self.word_likelihoods = {
            cls: {
                word: (word_counts_per_class[cls][word] + self.smoothing) / 
                      (total_words_per_class[cls] + self.smoothing * vocab_size)
                for word in self.vocab 
            }
            for cls in self.classes
        }

    def predict(self, X: List[List[str]]) -> List[str]:
        """
        Predicts class labels for a batch of documents.
        Args:
            X: List of tokenized documents
        Returns:
            List of predicted class labels
        """
        predictions = []
        
        for doc in X:
            scores = {}
            for cls in self.classes:
                log_prob = np.log(self.class_priors[cls])
                for word in doc:
                    if word in self.vocab:
                        # Use log(P(w|y)) to prevent underflow
                        log_prob += np.log(self.word_likelihoods[cls].get(word, 1e-3))
                scores[cls] = log_prob
                
            # Choose class with max log-probability
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
        
        return predictions

    def score(self, X: List[List[str]], y: List[str]) -> float:
        """
        Computes accuracy on the given dataset.
        """
        preds = self.predict(X)
        return np.mean(np.array(preds) == np.array(y))

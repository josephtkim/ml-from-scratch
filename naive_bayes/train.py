from typing import List
from model import NaiveBayesTextClassifier

def preprocess(texts: List[str]) -> List[List[str]]:
    """
    Tokenizes and cleans raw texts.
    """
    return [text.lower().split() for text in texts]

def load_dummy_dataset():
    """
    Loads a small dummy dataset for spam/ham classification.
    Returns:
        X_train, y_train: training inputs and labels
        X_test, y_test: test inputs and labels
    """
    # Simple dataset (can be expanded)
    texts = [
        "Win a free iPhone now",
        "Call this number to claim your prize",
        "Let's grab lunch tomorrow",
        "Are we still meeting later?",
        "Free money waiting for you",
        "Congrats, you won the lottery",
        "Can you send me the report?",
        "Don't forget the meeting at 10am",
        "Are you available for a meeting next week?",
        "Please review the attached file",
    ]

    labels = [
        "spam", "spam",
        "ham", "ham",
        "spam", "spam",
        "ham", "ham",
        "ham", "ham"
    ]

    # More balanced split
    X_train_raw = texts[:8]
    y_train = labels[:8]
    X_test_raw = texts[8:]
    y_test = labels[8:]

    return X_train_raw, y_train, X_test_raw, y_test

def train_and_evaluate():
    # Load and preprocess data
    X_train_raw, y_train, X_test_raw, y_test = load_dummy_dataset()
    X_train = preprocess(X_train_raw)
    X_test = preprocess(X_test_raw)

    # Initialize model
    model = NaiveBayesTextClassifier(smoothing=1.0)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClass Priors:")
    print(model.class_priors)

    print("\nVocabulary Size:", len(model.vocab))
    print("Vocabulary Sample:", list(model.vocab)[:10])

    # Print predictions for manual inspection
    preds = model.predict(X_test)
    for doc, pred, actual in zip(X_test_raw, preds, y_test):
        print(f"Text: \"{doc}\" → Predicted: {pred}, Actual: {actual}")

if __name__ == "__main__":
    train_and_evaluate()

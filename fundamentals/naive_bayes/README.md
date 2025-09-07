# 🧠 Naive Bayes for Text Classification — Background

---

### 🎯 Goal

To classify text documents into predefined categories (e.g., **spam/ham**, **positive/negative sentiment**) by computing the **probability of a class given input features** (typically words or tokens in text).

---

### 📉 Loss / Objective

Naive Bayes uses **Maximum Likelihood Estimation (MLE)** or **Maximum A Posteriori (MAP)** to find the most probable class:

$$
\hat{y} = \arg\max_y \; P(y \mid x_1, x_2, ..., x_n)
$$

With the **naive assumption** that features (words) are conditionally independent given the class:

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

To avoid numerical underflow (especially when multiplying many small probabilities), we compute in **log-space**:

$$
\log P(y \mid \mathbf{x}) \propto \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y)
$$

During inference, the model computes this sum of log-probabilities for each class and selects the class with the highest total:

$$
\hat{y} = \arg\max_y \left[ \log P(y) + \sum_{i=1}^{n} \log P(x_i \mid y) \right]
$$

This is the **core scoring mechanism** implemented in prediction, and it serves as a proxy for minimizing the following loss:

$$
\mathcal{L} = - \log P(y \mid \mathbf{x})
$$

---

### 🧠 What's Optimized

Naive Bayes does **not** rely on iterative gradient-based training. Instead, it:

- Estimates **class prior probabilities** \( P(y) \)
- Estimates **word likelihoods** \( P(x_i \mid y) \)

These are computed using **frequency counts** from the training data, often with **Laplace smoothing** to avoid assigning zero probability to unseen words.

---


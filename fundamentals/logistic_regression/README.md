# 📘 Logistic Regression: Overview

---

### 🎯 Goal

To model the **probability** that a binary outcome variable $y \in \{0,1\}$ occurs given input features $X$.

Output is interpreted as:
$$
\hat{y} = P(y=1 \mid X)
$$

Given:
- $X \in \mathbb{R}^{n \times d}$: input features (rows = samples, columns = features)
- $w \in \mathbb{R}^{d}$: weights
- $b \in \mathbb{R}$: bias (scalar)

1. **Linear combination**:
   $$
   z = Xw + b
   $$

2. **Apply sigmoid function** to map to probability:
   $$
   \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
   $$

- This maps any real number $z$ into the range $(0, 1)$
- $\hat{y}$ represents the **probability** of the sample belonging to class 1

---

### 📉 Loss / Objective

- **Binary Cross Entropy (Log Loss)**:
  $$
  \mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  $$

---

### 🧠 What's Optimized

- The **weights** $w$ and **bias** $b$
- Optimized to **minimize log loss** via gradient descent or similar optimization methods

---
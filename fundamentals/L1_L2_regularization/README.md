# 📘 L1 and L2 Regularization — Background

---

### 🎯 Goal

To **prevent overfitting** by discouraging overly complex models.  
Regularization improves generalization by penalizing large weights.

---

### 📉 Loss / Objective

Assuming a linear model:

$$
\hat{y} = Xw + b
$$

And a base loss (e.g., Mean Squared Error):

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### 🔹 Ridge Regression (L2)

Adds an L2 penalty (squared weights):

$$
\mathcal{L}_{\text{ridge}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_j w_j^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \|w\|_2^2
$$

- Encourages **smaller weights**.
- No sparsity (weights shrink but rarely reach zero).

### 🔸 Lasso Regression (L1)

Adds an L1 penalty (absolute weights):

$$
\mathcal{L}_{\text{lasso}} = \mathcal{L}_{\text{MSE}} + \lambda \sum_j |w_j| = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \|w\|_1
$$

- Encourages **sparsity** (some weights exactly zero).
- Performs **feature selection**.

---

### 🧠 What's Optimized

We minimize the **regularized loss function** with respect to:

- **Weights** \( w \)
- **Bias** \( b \)

The choice of penalty affects:

| Type   | Encourages         | Common Solver    |
|--------|--------------------|------------------|
| L2     | Small weights      | Gradient descent |
| L1     | Sparse weights     | Subgradient / Coordinate descent |

---


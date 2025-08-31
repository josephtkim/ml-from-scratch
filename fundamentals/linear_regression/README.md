# 📘 Linear Regression: Overview

---

### 🎯 Goal

To model the **linear relationship** between input features $X$ and a continuous target variable $y$, enabling prediction of $y$ from new inputs.

---

### ✅ Assumptions

1. **Linearity**: $y$ is a linear combination of input features.
2. **Independence**: Residuals (errors) are independent.
3. **Homoscedasticity**: Constant variance of residuals.
4. **Normality of errors**: Residuals are normally distributed (for inference).
5. **No multicollinearity**: Features should not be highly correlated.

---

### 📉 Loss / Objective

**Mean Squared Error (MSE):**

$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

Minimizing the MSE ensures the model learns coefficients that result in predictions $\hat{y}_i$ close to actual values.

---

### 🧠 What's Optimized

The **model parameters** $\theta$ (weights) are optimized by solving the **normal equations** or using **gradient descent**, aiming to minimize the MSE loss.

---

### ⚠️ 5 Common Pitfalls / Edge Cases

1. **Multicollinearity**: Leads to unstable coefficient estimates.
2. **Outliers**: Can disproportionately affect the fit.
3. **Non-linearity**: If the true relationship is nonlinear, performance degrades.
4. **High-dimensional data (n < p)**: No unique solution without regularization.
5. **Heteroscedasticity**: Violates assumptions, affects standard errors and inference.

---

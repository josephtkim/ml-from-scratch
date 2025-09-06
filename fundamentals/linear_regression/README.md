# 📘 Linear Regression: Overview

---

### 🎯 Goal

To model the **linear relationship** between input features $X$ and a continuous target variable $y$, enabling prediction of $y$ from new inputs.

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
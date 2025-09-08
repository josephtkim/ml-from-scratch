# 📘 Support Vector Machines — Background

---

### 🎯 Goal

To find the optimal hyperplane that separates classes **with the largest possible margin**, improving generalization on unseen data.

---

### 📉 Loss / Objective

SVMs optimize a **hinge loss** combined with an **L2 regularization** term (in the soft-margin case):

\[
\min_{\mathbf{w}, b} \ \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))
\]

- **Hard margin**: assumes perfect linear separability.
- **Soft margin**: allows for margin violations, controlled by regularization parameter **C**.

---

### 🧠 What's Optimized

- **Model parameters**: weight vector **w** and bias **b** defining the decision boundary.
- **Goal**: maximize the margin while minimizing misclassification penalty.
- **Optimization**: convex problem; solved via **gradient descent**, **quadratic programming**, or **SMO**.

---


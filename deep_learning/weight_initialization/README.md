# 📘 Weight Initialization — Background

---

### 🎯 Goal

To initialize weights so that **variance of activations and gradients remains stable** across layers, avoiding vanishing/exploding issues.

For a given layer \( l \):

- Let  
  \[
  x^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
  \]
- We want:  
  \[
  \text{Var}[x^{(l)}] \approx \text{Var}[a^{(l-1)}]
  \quad \text{and} \quad
  \text{Var}[\nabla x^{(l)}] \approx \text{Var}[\nabla a^{(l)}]
  \]

---

### 📉 Loss / Objective

Weight initialization doesn’t affect the form of the loss function. Typical supervised losses apply:

- **Classification**:  
  \[
  \mathcal{L} = \text{CrossEntropy}(y, \hat{y})
  \]
- **Regression**:  
  \[
  \mathcal{L} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  \]

But initialization **impacts how effectively optimization algorithms minimize these losses**.

---

### 🧠 What's Optimized

Weights \( W^{(l)} \) and biases \( b^{(l)} \) are trained, but **initialization sets their starting values**.

**Xavier (Glorot) Initialization** — for **tanh/sigmoid** activations:  
\[
W^{(l)} \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}} + n_{\text{out}}} \right)
\quad \text{or} \quad
\mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}} \right]
\]

**He Initialization** — for **ReLU** activations:  
\[
W^{(l)} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}} \right)
\quad \text{or} \quad
\mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}} \right]
\]

Where:
- \( n_{\text{in}} \): Number of inputs to the layer  
- \( n_{\text{out}} \): Number of outputs from the layer

These schemes help maintain a stable signal as it flows **forward (activations)** and **backward (gradients)** through the network.

---

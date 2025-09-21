# 📘 Weight Initialization — Background

---

### 🎯 Goal

To initialize weights so that **variance of activations and gradients remains stable** across layers, avoiding vanishing or exploding values.

For a given layer `l`:

- Let:  
  `x^(l) = W^(l) * a^(l-1) + b^(l)`

- We want:  
  `Var[x^(l)] ≈ Var[a^(l-1)]`  
  `Var[∇x^(l)] ≈ Var[∇a^(l)]`

---

### 📉 Loss / Objective

Weight initialization doesn’t affect the form of the loss function. Typical supervised losses apply:

- **Classification**:  
  `L = CrossEntropy(y, ŷ)`

- **Regression**:  
  `L = (1/n) * sum_i (y_i - ŷ_i)^2`

But initialization **impacts how effectively optimization algorithms minimize these losses** — especially in deep networks.

---

### 🧠 What's Optimized

Weights `W^(l)` and biases `b^(l)` are trained, but **initialization sets their starting values**.

**Xavier (Glorot) Initialization** — for **tanh/sigmoid** activations:  
- `W^(l) ~ N(0, 1 / (n_in + n_out))`  
  or  
- `W^(l) ~ U[-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))]`

**He Initialization** — for **ReLU** activations:  
- `W^(l) ~ N(0, 2 / n_in)`  
  or  
- `W^(l) ~ U[-sqrt(6 / n_in), sqrt(6 / n_in)]`

Where:
- `n_in`  = number of inputs to the layer  
- `n_out` = number of outputs from the layer

These schemes help maintain a stable signal as it flows **forward (activations)** and **backward (gradients)** through the network.

---

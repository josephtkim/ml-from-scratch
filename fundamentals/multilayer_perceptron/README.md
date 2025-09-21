# 📘 Multilayer Perceptron — Background

---

### 🎯 Goal

To approximate complex **non-linear functions** by stacking fully connected layers with learnable weights and non-linear activation functions.

Given:
- Input `x ∈ ℝᵈ`
- Layer sizes: `[n₀, n₁, ..., n_L]` where `n₀ = d` (input dim), `n_L` = output dim

For each layer `ℓ = 1, ..., L`:
1. **Linear transformation**:

   ```
   z[ℓ] = W[ℓ] · a[ℓ−1] + b[ℓ]
   ```

2. **Apply non-linearity**:

   ```
   a[ℓ] = σ(z[ℓ])
   ```

Where:
- `a[0] = x` (the input)
- `W[ℓ] ∈ ℝⁿˡ × ⁿˡ⁻¹` (weights)
- `b[ℓ] ∈ ℝⁿˡ` (biases)
- `σ`: activation function (e.g. ReLU or softmax)

The final output `ŷ = a[L]` is the prediction.

---

### 📉 Loss / Objective

For **classification** (e.g. softmax output), use **cross-entropy loss**:

```
L = −∑ yᵢ log(ŷᵢ)
```

For **regression**, use **mean squared error (MSE)**:

```
L = (1/N) ∑ (yᵢ − ŷᵢ)²
```

---

### 🧠 What's Optimized

The model learns:
- Weights `W[ℓ]` and biases `b[ℓ]` for all layers
- Optimization is done by minimizing the loss `L` using **gradient descent**

Using **backpropagation**, gradients are computed recursively:

- For the **output layer**:

  ```
  δ[L] = ∇ₐ L ⊙ σ′(z[L])
  ```

- For **hidden layers**:

  ```
  δ[ℓ] = (W[ℓ+1]ᵀ · δ[ℓ+1]) ⊙ σ′(z[ℓ])
  ```

Parameter updates (for all layers `ℓ`):

```
W[ℓ] ← W[ℓ] − η · ∂L/∂W[ℓ]
b[ℓ] ← b[ℓ] − η · ∂L/∂b[ℓ]
```

Where:
- `η` is the learning rate
- `δ[ℓ]` is the error signal at layer `ℓ`
- `⊙` is element-wise (Hadamard) product

---

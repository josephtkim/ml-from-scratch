# 📘 Multilayer Perceptron — Background

---

### 🎯 Goal

To approximate complex **non-linear functions** by stacking fully connected layers with learnable weights and non-linear activation functions.

Given:
- Input $x \in \mathbb{R}^d$
- Layer sizes: $[n_0, n_1, \dots, n_L]$ where $n_0 = d$ (input dim), $n_L$ = output dim

For each layer $\ell = 1, \dots, L$:
1. **Linear transformation**:
   $$
   z^{[\ell]} = W^{[\ell]} a^{[\ell - 1]} + b^{[\ell]}
   $$
2. **Apply non-linearity**:
   $$
   a^{[\ell]} = \sigma(z^{[\ell]})
   $$

Where:
- $a^{[0]} = x$ (input)
- $W^{[\ell]} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ (weights)
- $b^{[\ell]} \in \mathbb{R}^{n_\ell}$ (biases)
- $\sigma$: activation function (e.g. ReLU, softmax)

The final output $\hat{y} = a^{[L]}$ is the prediction.

---

### 📉 Loss / Objective

For **classification** (e.g. with softmax output), use **cross-entropy loss**:
$$
\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

For **regression**, use **mean squared error (MSE)**:
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

---

### 🧠 What's Optimized

The model learns:
- Weights $W^{[\ell]}$ and biases $b^{[\ell]}$ for all layers $\ell$
- Optimization is done by minimizing the loss $\mathcal{L}$ via gradient descent

Using **backpropagation**, gradients are computed recursively:
- For output layer:
  $$
  \delta^{[L]} = \nabla_{a^{[L]}} \mathcal{L} \odot \sigma'(z^{[L]})
  $$
- For hidden layers:
  $$
  \delta^{[\ell]} = \left( W^{[\ell + 1]^\top} \delta^{[\ell + 1]} \right) \odot \sigma'(z^{[\ell]})
  $$

Parameter updates:
$$
W^{[\ell]} \leftarrow W^{[\ell]} - \eta \frac{\partial \mathcal{L}}{\partial W^{[\ell]}}, \quad 
b^{[\ell]} \leftarrow b^{[\ell]} - \eta \frac{\partial \mathcal{L}}{\partial b^{[\ell]}}
$$

Where:
- $\eta$ is the learning rate
- $\delta^{[\ell]}$ is the error signal at layer $\ell$
- $\odot$ is element-wise (Hadamard) product

---


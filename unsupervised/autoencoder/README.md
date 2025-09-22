# 📘 K-Means: Overview

---

### 🎯 Goal

To learn a **compressed representation** (encoding) of input data and then reconstruct the original input from it.  

Common use cases:
- Dimensionality reduction  
- Denoising  
- Representation learning  

---

### 📉 Loss / Objective

Minimize **reconstruction loss** between input \( x \in \mathbb{R}^d \) and reconstruction \( \hat{x} \in \mathbb{R}^d \).  

Typical loss (mean squared error):

$$
\mathcal{L}(x, \hat{x}) = \|x - \hat{x}\|^2
$$

- If reconstruction is **perfect**, loss is 0.  
- Larger reconstruction errors increase the loss quadratically.  

---

### 🧠 What's Optimized

An autoencoder jointly learns:  

- **Encoder** \( f_{\theta}(x) \): maps input \( x \) to latent representation \( z \).  
- **Decoder** \( g_{\phi}(z) \): reconstructs \( x \) from latent \( z \).  

Parameters \( \theta, \phi \) are optimized via gradient descent to minimize reconstruction error.

---

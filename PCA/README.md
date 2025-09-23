# 📘 PCA: Overview

---

### 🎯 Goal

Reduce high-dimensional data to a lower-dimensional subspace while preserving the most **important structure** — specifically, the directions of **maximum variance**.

Given:

- **Input data matrix**:  
  `X ∈ ℝⁿˣᵈ`  
  where `n` is the number of samples and `d` is the number of features

We aim to find a set of **orthonormal vectors** (principal axes):

- `W ∈ ℝᵈˣᵏ`, with `k < d`

so that the data projected onto this new subspace retains the most variance.

---

### 📉 Loss / Objective

PCA solves the following optimization problem:

> **Minimize reconstruction error**:  
>  
>  `L = ||X - X_proj||²_F = ||X - X W Wᵀ||²_F`

Or equivalently:

> **Maximize variance along projections**:  
>  
>  maximize `Tr(Wᵀ Σ W)`  
>  subject to `Wᵀ W = I`

Where:

- `Σ = (1/n) Xᵀ X` is the empirical covariance matrix (after centering `X`)
- `W` consists of the top-`k` eigenvectors of `Σ`

Thus, PCA is both a **compression** and a **variance-preserving projection** method.

---

### 🧠 What's Optimized

PCA optimizes for a **projection matrix `W`** such that:

- The columns of `W` (principal components) are **orthonormal**  
  `Wᵀ W = I`
- The directions in `W` capture the **largest variance** in the dataset

This is solved by computing the **eigen-decomposition** of the covariance matrix.

---

### 🧮 Steps to Compute PCA

1. **Center the data**:  
   Subtract the mean of each feature from the data.  
   `X_centered = X - μ`

2. **Compute the covariance matrix**:  
   `Σ = (1/n) X_centeredᵀ X_centered`

3. **Eigen-decompose** the covariance matrix:  
   `Σ = V Λ Vᵀ`

4. **Sort eigenvalues** `λᵢ` in descending order

5. **Select top-`k` eigenvectors**:  
   `W ∈ ℝᵈˣᵏ`

6. **Project the data**:  
   `X_reduced = X_centered ⋅ W`

---

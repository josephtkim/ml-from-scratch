# 📘 PCA: Overview

---

### 🎯 Goal

Reduce high-dimensional data to a lower-dimensional subspace while preserving the most **important structure** — specifically, the directions of **maximum variance**.

Given:

- **Input data matrix**:  
  \[
  X \in \mathbb{R}^{n \times d}
  \]  
  where \( n \) is the number of samples and \( d \) is the number of features

We aim to find a set of **orthonormal vectors** (principal axes):

- \[
  W \in \mathbb{R}^{d \times k}, \quad \text{with } k < d
  \]

so that the data projected onto this new subspace retains the most variance.

---

### 📉 Loss / Objective

PCA solves the following optimization problem:

> **Minimize reconstruction error**:  
\[
\mathcal{L} = \|X - X_{proj}\|_F^2 = \|X - XWW^\top\|_F^2
\]

or equivalently,

> **Maximize variance along projections**:  
\[
\text{maximize } \operatorname{Tr}(W^\top \Sigma W), \quad \text{subject to } W^\top W = I
\]

Where:

- \( \Sigma = \frac{1}{n} X^\top X \) is the empirical covariance matrix (after centering \( X \))
- \( W \) consists of the top-k eigenvectors of \( \Sigma \)

Thus, PCA is both a **compression** and a **variance-preserving projection** method.

---

### 🧠 What's Optimized

PCA optimizes for a **projection matrix \( W \)** such that:

- The columns of \( W \) (principal components) are **orthonormal**  
  \[
  W^\top W = I
  \]
- The directions in \( W \) capture the **largest variance** in the dataset
- Formally, this is solved by computing the **eigen-decomposition** of the covariance matrix:

### 🧮 Steps to Compute PCA:
1. **Center the data**:  
   Subtract the mean of each feature from the data.
   \[
   X_{centered} = X - \mu
   \]
2. **Compute the covariance matrix**:  
   \[
   \Sigma = \frac{1}{n} X_{centered}^\top X_{centered}
   \]
3. **Eigen-decompose** \( \Sigma \):  
   \[
   \Sigma = V \Lambda V^\top
   \]
4. **Sort eigenvalues** \( \lambda_i \) in descending order
5. **Select top-k eigenvectors** \( W \in \mathbb{R}^{d \times k} \)
6. **Project the data**:
   \[
   X_{reduced} = X_{centered} \cdot W
   \]

---

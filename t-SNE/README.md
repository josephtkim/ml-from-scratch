# 📘 t-SNE: Overview

---

### 🎯 Goal

To visualize high-dimensional data in 2D or 3D while preserving **local structure** — that is, **similar points stay close**, while **dissimilar points stay apart**.

---

### 📉 Loss / Objective

t-SNE minimizes the **Kullback–Leibler (KL) divergence** between:

- High-dimensional similarities:  
  $$ P_{ij} $$  
- Low-dimensional similarities:  
  $$ Q_{ij} $$

The loss function is:

$$
\mathcal{L}_{\text{t-SNE}} = \sum_{i \ne j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
$$

- \( P_{ij} \) is computed using a **Gaussian kernel** in the original space.
- \( Q_{ij} \) uses a **Student’s t-distribution** in the low-dimensional space (e.g., 2D), helping avoid the **crowding problem**.

---

### 🧠 What's Optimized

t-SNE optimizes the **low-dimensional coordinates**:

$$
Y \in \mathbb{R}^{n \times d'}
$$

where \( d' = 2 \) or \( 3 \), using **gradient descent** to minimize KL divergence:

- High-dimensional similarities \( P_{ij} \) are computed once.
- Low-dimensional coordinates \( Y \) are iteratively updated so that \( Q_{ij} \approx P_{ij} \).

---
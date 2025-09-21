# 📘 K-Means: Overview

---

### 🎯 Goal

To partition data into **K clusters** such that each data point belongs to the cluster with the **nearest centroid**.  
Used in **unsupervised learning** to discover structure in data.

---

### 📉 Loss / Objective

Minimize **intra-cluster variance** (a.k.a. **distortion** or **within-cluster sum of squares**):

$$
\mathcal{L} = \sum_{i=1}^{n} \|x_i - \mu_{c_i}\|^2
$$

- $x_i$: data point  
- $c_i$: index of cluster assignment for $x_i$
- $\mu_{c_i}$: centroid of assigned cluster  

---

### 🧠 What's Optimized

- Cluster **centroids** $\mu_k \in \mathbb{R}^d$
- Cluster **assignments** $c_i \in \{1, \dots, K\}$

Optimization is done via **coordinate descent**:  
1. **Assignment step**: fix centroids, assign points to closest centroid  
2. **Update step**: fix assignments, recompute centroids as means of assigned points  

---

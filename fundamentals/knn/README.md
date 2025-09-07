# 📘 K-Nearest Neighbors: Overview

---

### 🎯 Goal

To classify a new sample \(x\) by looking at the **K most similar (nearest) training samples** and assigning the most frequent label among them.  
KNN can also be used for regression by averaging the target values of neighbors.

---

### 📉 Loss / Objective

KNN is **non-parametric** and **instance-based**, meaning:

- There is **no explicit loss function** minimized during training.
- There is **no training phase** (other than storing the dataset).
- The "optimization" happens at **inference time** (i.e., finding neighbors and aggregating their labels).

However, the classification goal can be loosely thought of as **minimizing the misclassification rate** over the training set if evaluated retrospectively.

---

### 🧠 What's Optimized

- **Nothing is learned** during training.
- At test time, the algorithm performs:
  - Distance computation (e.g., Euclidean)
  - Neighbor selection (top K)
  - Majority vote or weighted vote 
  
- Choice of hyperparameters and design:
  - \(K\) (number of neighbors)
  - Distance metric (Euclidean, Manhattan, cosine, etc.)

---
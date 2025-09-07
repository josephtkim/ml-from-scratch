# 📘 Decision Trees for Classification — Background

---

### 🎯 Goal

To predict the class label of input data by learning a sequence of decision rules based on feature values — represented as a tree structure. Each **internal node** is a decision (e.g., `x_i > t`), and each **leaf node** holds a predicted class.

---

### 📉 Loss / Objective

Most common loss functions used in classification:

- **Gini Impurity**  
  $$
  \text{Gini}(D) = 1 - \sum_{c} p(c)^2
  $$

- **Entropy (Information Gain)**  
  $$
  H(D) = -\sum_{c} p(c) \log_2 p(c)
  $$

The goal is to **recursively split** the dataset at each node to **maximize information gain** (i.e., reduce impurity).

---

### 🧠 What's Optimized

Decision Trees greedily find the **best feature** and **threshold** that result in the **largest impurity reduction**.

Unlike models that rely on gradient descent, Decision Trees use a **non-iterative, recursive partitioning** approach. Specifically, they:

- Evaluate all candidate splits
- Select the split that minimizes the impurity of the resulting children
- Recurse until a stopping condition is met (e.g., max depth, pure leaves, or minimum number of samples per split)

---


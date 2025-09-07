# 📘 Decision Trees for Classification — Background

---

### 🎯 Goal

To improve classification accuracy and reduce overfitting by combining the predictions of multiple decision trees trained on different subsets of data and features - an ensemble method known as **bagging** (Bootstrap Aggregating).

---

### 📉 Loss / Objective

Each individual decision tree in the forest minimizes a standard impurity-based loss (e.g., **Gini Impurity** or **Entropy**).

The **overall Random Forest** does not have a single global loss - instead, its **objective is to reduce variance** through aggregation of diverse decision trees.

---

### 🧠 What's Optimized

- **Tree-level**: Each tree greedily optimizes impurity reduction (like a standard Decision Tree).
- **Forest-level**: Random Forests optimize for:
  - **Low correlation** between trees (via random feature selection)
  - **High individual accuracy** (via strong learners)
  - **Ensemble prediction** via **majority voting** (classification)

---


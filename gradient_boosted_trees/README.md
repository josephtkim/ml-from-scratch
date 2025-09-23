# 📘 Gradient Boosted Trees — Background

---

### 🎯 Goal

To build a strong ensemble model by **sequentially adding trees** that correct the errors of the previous ensemble. Each new tree is trained to **reduce the residual errors** of the current model.

---

### 📉 Loss / Objective

- The model minimizes a **global differentiable loss function** over the training set.
- Common examples:
  - **MSE (Mean Squared Error)** for regression
  - **Log Loss (Cross-Entropy)** for classification

---

### 🧠 What's Optimized

- **At each iteration**, the model fits a new tree to the **negative gradient** of the loss function (i.e., a **pseudo-residual**).
- Each tree minimizes the **residual loss**, and the ensembble prediction is udpated additively. 

---


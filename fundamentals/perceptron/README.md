# 📘 Perceptron — Background

---

### 🎯 Goal

To learn a **linear decision boundary** that separates data into two classes (typically binary classification, e.g., 0 or 1).

Given:
- \( X \in \mathbb{R}^{n \times d} \): input features  
- \( w \in \mathbb{R}^d \): weights  
- \( b \in \mathbb{R} \): bias  

The model computes a **linear combination**:

$$
z = Xw + b
$$

Then applies a **step activation function**:

$$
\hat{y} = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

### 📉 Loss / Objective

The **Perceptron loss** only penalizes misclassified samples.

For each sample \( (x_i, y_i) \), where \( y_i \in \{-1, +1\} \), the loss is:

$$
\mathcal{L} = \sum_{i=1}^{n} \max(0, -y_i (w^\top x_i + b))
$$

- If the sample is **correctly classified** (margin ≥ 0), there’s no loss.  
- If **misclassified**, loss increases linearly with the margin violation.

---

### 🧠 What's Optimized

- The **weights** \( w \) and **bias** \( b \)  
- Updated using the **Perceptron update rule**:

If a sample is **misclassified**, update:

$$
w \leftarrow w + \alpha \cdot y_i x_i
$$

$$
b \leftarrow b + \alpha \cdot y_i
$$

- \( \alpha \) is the learning rate  
- This pushes the decision boundary **toward correctly classifying** the sample

---


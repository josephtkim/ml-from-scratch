# 📘 Multinomial Logistic Regression: Overview

---

### 🎯 Goal

To model the probability that a sample belongs to one of **K > 2** discrete classes, given input features \( X \).  

Output is interpreted as:  
$$\hat{y}_i = P(y = k \mid X_i) \quad \text{for } k \in \{1, 2, \dots, K\}$$

- $$X \in \mathbb{R}^{n \times d}$$: input features (rows = samples, columns = features)  
- $$W \in \mathbb{R}^{K \times d}$$: weight matrix (one weight vector per class)  
- $$b \in \mathbb{R}^{K}$$: bias vector  

Linear combination for each class:  
$$z_i = W X_i + b \quad \text{for each sample } i$$

Apply **softmax** function to map to class probabilities:  

$$\hat{y}_{ik} = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}$$


This ensures:
- $$\hat{y}_{ik} \in (0, 1)$$ 
- $$\sum_k \hat{y}_{ik} = 1$$

---

### 📉 Loss / Objective

**Multiclass Cross-Entropy Loss**:  
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

where:
- $$y_{ik} = 1$$ if true class of sample $i$ is $k$, else 0  
- $$\hat{y}_{ik}$$ is the predicted probability for class $k$

---

### 🧠 What's Optimized

The weight matrix $W$ and bias vector $b$
These are learned to **minimize the cross-entropy loss** using gradient-based methods like **Stochastic Gradient Descent (SGD)** or **Adam**.

---

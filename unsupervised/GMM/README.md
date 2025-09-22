# 📘 Gaussian Mixture Models (EM): Overview

---

### 🎯 Goal

Model data as a **mixture of multiple Gaussian distributions**, where each Gaussian corresponds to a latent cluster.  
Unlike k-means (which uses hard assignments), GMMs use **soft assignments** — each sample belongs to clusters with some probability.

---

### 📉 Loss / Objective

We want to **maximize the likelihood** of the observed data under the mixture model:

$$
\mathcal{L}(\theta) = \sum_{i=1}^{n} \log \left( \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)
$$

Where:

- $$\pi_k$$ are the **mixing coefficients** (weights), subject to:

  $$
  \sum_{k=1}^{K} \pi_k = 1
  $$

- $$\mu_k$$ is the **mean vector** of the \(k\)-th Gaussian
- $$\Sigma_k$$ is the **covariance matrix** of the \(k\)-th Gaussian

---

### 🧠 What's Optimized

Parameters learned:

- **Mixing weights**: $$\pi_k$$  
- **Means**: $$\mu_k$$  
- **Covariances**: $$\Sigma_k$$

Optimization is performed using the **Expectation–Maximization (EM)** algorithm:

- **E-step**: Estimate **responsibilities** — the posterior probability that data point $$x_i$$ belongs to cluster $$k$$:

  $$
  \gamma_{ik} = \frac{ \pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k) }{ \sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j) }
  $$

- **M-step**: Update the parameters to maximize the expected log-likelihood given the current responsibilities:

  $$
  \pi_k = \frac{1}{n} \sum_{i=1}^{n} \gamma_{ik}
  $$

  $$
  \mu_k = \frac{ \sum_{i=1}^{n} \gamma_{ik} x_i }{ \sum_{i=1}^{n} \gamma_{ik} }
  $$

  $$
  \Sigma_k = \frac{ \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top }{ \sum_{i=1}^{n} \gamma_{ik} }
  $$

This EM procedure repeats until the **log-likelihood converges**, typically measured by:

$$
|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \varepsilon
$$

where $$\varepsilon$$ is a small threshold (e.g., $$10^{-4}$$).

---
# Mathematical Foundations

## 1. Multi-Layer Perceptron (XOR)
Forward pass with Sigmoid activation $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Loss formulation via Binary Cross-Entropy (BCE):

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

## 2. Scaled Dot-Product Causal Attention

$$Attention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

Where $M$ is the causal mask matrix:

$$M_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

## 3. Autoregressive Sampling

Softmax with Temperature $T$:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Top-P (Nucleus) selection set $V^{(p)}$:

$$\sum_{x \in V^{(p)}} P(x) \ge p$$

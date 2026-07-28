import numpy as np
from typing import Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along a specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class CausalMultiHeadAttention:
    """Causal Multi-Head Self-Attention built from scratch in pure NumPy.

    Ensures tokens can only attend to previous tokens and themselves via a causal lower-triangular mask.
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Weight matrices initialized with Xavier scaling
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)."""
        batch_size, seq_len, _ = x.shape
        reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return reshaped.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)."""
        batch_size, num_heads, seq_len, d_k = x.shape
        transposed = x.transpose(0, 2, 1, 3)
        return transposed.reshape(batch_size, seq_len, num_heads * d_k)

    def forward(
        self, x: np.ndarray, return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Forward pass for causal self-attention.

        Args:
            x: Input matrix of shape (batch_size, seq_len, d_model)
            return_attention: Whether to return attention weight matrix

        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = np.dot(x, self.W_q)  # (batch_size, seq_len, d_model)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Split into multiple heads
        Q = self._split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Scaled Dot-Product Attention: Scores = (Q @ K^T) / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)

        # Apply Causal Mask (lower-triangular matrix of ones)
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        scores = np.where(causal_mask == 0, -1e9, scores)

        # Compute softmax over key dimension
        attn_weights = softmax(scores, axis=-1)

        # Weighted sum of values: Context = Scores @ V
        context = np.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)

        # Combine heads and apply final projection
        combined = self._combine_heads(context)  # (batch, seq_len, d_model)
        output = np.dot(combined, self.W_o)

        if return_attention:
            return output, attn_weights
        return output, None


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    x = np.random.randn(batch_size, seq_len, d_model)

    mha = CausalMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    out, attn = mha.forward(x, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention Weights shape: {attn.shape}")
    print("\nCausal Mask Verification (Upper triangle should be 0.0):")
    print(np.round(attn[0, 0], 3))

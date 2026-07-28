import numpy as np
from typing import Tuple
from src.transformer.attention import CausalMultiHeadAttention


class LayerNorm:
    """Layer Normalization built from scratch."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class PositionWiseFFN:
    """Position-wise Feed-Forward Network with ReLU activation."""

    def __init__(self, d_model: int, d_ff: int):
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))
        self.W1 = np.random.randn(d_model, d_ff) * scale1
        self.b1 = np.zeros((1, 1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model) * scale2
        self.b2 = np.zeros((1, 1, d_model))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Layer 1 + ReLU
        h = np.maximum(0, np.dot(x, self.W1) + self.b1)
        # Layer 2
        return np.dot(h, self.W2) + self.b2


class TransformerDecoderBlock:
    """Single Decoder-Only Transformer Block using Pre-LayerNorm architecture."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attn = CausalMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = PositionWiseFFN(d_model=d_model, d_ff=d_ff)
        self.ln1 = LayerNorm(d_model=d_model)
        self.ln2 = LayerNorm(d_model=d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with residual connections and Pre-LN normalization."""
        # 1. Causal Attention Sub-layer with residual connection
        attn_out, _ = self.attn.forward(self.ln1.forward(x))
        x = x + attn_out

        # 2. Feed-Forward Sub-layer with residual connection
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = x + ffn_out

        return x


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 8, 32, 4, 128
    x = np.random.randn(batch_size, seq_len, d_model)

    block = TransformerDecoderBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    out = block.forward(x)

    print(f"Decoder Block Input: {x.shape}")
    print(f"Decoder Block Output: {out.shape}")

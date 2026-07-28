import pytest
import numpy as np
from src import (
    CausalMultiHeadAttention,
    LayerNorm,
    PositionWiseFFN,
    TransformerDecoderBlock,
)


def test_causal_attention_shapes():
    batch, seq, d_model, heads = 2, 6, 16, 4
    x = np.random.randn(batch, seq, d_model)

    attn = CausalMultiHeadAttention(d_model=d_model, num_heads=heads)
    out, weights = attn.forward(x, return_attention=True)

    assert out.shape == (batch, seq, d_model)
    assert weights.shape == (batch, heads, seq, seq)


def test_causal_mask_enforcement():
    batch, seq, d_model, heads = 1, 4, 8, 2
    x = np.random.randn(batch, seq, d_model)

    attn = CausalMultiHeadAttention(d_model=d_model, num_heads=heads)
    _, weights = attn.forward(x, return_attention=True)

    # Upper triangular elements (excluding main diagonal) must be strictly 0.0
    for head in range(heads):
        upper_triangle = np.triu(weights[0, head], k=1)
        np.testing.assert_array_equal(upper_triangle, np.zeros((seq, seq)))


def test_layer_norm():
    batch, seq, d_model = 2, 5, 12
    x = np.random.randn(batch, seq, d_model) * 10.0 + 5.0

    ln = LayerNorm(d_model=d_model)
    normed = ln.forward(x)

    # Mean should be close to 0 and variance close to 1 across the feature axis
    means = np.mean(normed, axis=-1)
    vars = np.var(normed, axis=-1)

    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(vars, 1.0, atol=1e-3)


def test_transformer_decoder_block():
    batch, seq, d_model, heads, d_ff = 3, 7, 32, 4, 64
    x = np.random.randn(batch, seq, d_model)

    block = TransformerDecoderBlock(d_model=d_model, num_heads=heads, d_ff=d_ff)
    out = block.forward(x)

    assert out.shape == (batch, seq, d_model)
    assert not np.isnan(out).any()

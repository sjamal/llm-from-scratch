# System Architecture

## Component Overview

             +-----------------------+
             |  Raw Text Input Data  |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             |  BPETokenizer (Sub)   |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             | Token ID Sequence (X) |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             | Causal Attention Block|
             | (Masked Multi-Head)   |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             | Transformer Decoder   |
             | (Pre-LN + FFN Stack)  |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             | AutoregressiveSampler |
             | (Temp/Top-K/Top-P)    |
             +-----------+-----------+
                         |
                         v
             +-----------------------+
             | Next Token Prediction |
             +-----------------------+

## Module Specifications

1. **`src/rule_based`**: Eliza pattern-matching chatbot using regex transform rules.
2. **`src/neural_net`**: Multi-Layer Perceptron (MLP) trained on XOR using pure NumPy backpropagation.
3. **`src/tokenizer`**: Byte-Pair Encoding (BPE) subword algorithm built from scratch.
4. **`src/embeddings`**: Skip-Gram Word2Vec implementation with cross-entropy loss.
5. **`src/transformer`**: Scaled Dot-Product Causal Attention and Pre-LN Transformer Decoder Block.
6. **`src/generation`**: Temperature, Top-K, and Top-P (Nucleus) sampling mechanisms.



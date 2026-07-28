# Project Roadmap & Technical Milestones

## Phase 1: Historical & Math Foundations
- [x] **src/rule_based**: Implement Eliza-style regex matching and conversation loops.
- [x] **src/neural_net**: Build single vs. multi-layer perceptron to solve XOR using pure backpropagation in NumPy.

## Phase 2: Representation & Tokenization
- [x] **src/tokenizer**: Build a Byte-Pair Encoding (BPE) algorithm from scratch to learn subword vocabularies.
- [x] **src/embeddings**: Train Skip-gram Word2Vec vectors to capture semantic space and vector operations.

## Phase 3: Transformer Core
- [x] **src/transformer/attention.py**: Scaled dot-product attention with causal mask ($Q, K, V$).
- [x] **src/transformer/decoder.py**: Multi-head attention block, LayerNorm, and Feed-Forward Neural Network (FFN).

## Phase 4: Inference & Generation
- [ ] **src/generation/sampler.py**: Temperature, Top-K, and Top-P (Nucleus) sampling logic.
- [ ] **CLI Scripts**: Training and inference scripts for generating text on small datasets.

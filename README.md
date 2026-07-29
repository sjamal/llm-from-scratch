# LLM From Scratch 🧠

An end-to-end, reproducible educational project designed to demystify Large Language Models (LLMs). This repository breaks down modern AI architecture into bite-sized, inspectable Python components—from basic rule-based systems to autoregressive transformer generation.

---

## 📁 Repository Structure

```text
llm-from-scratch/
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── task.md
├── src/
│   ├── __init__.py
│   ├── rule_based/         # Eliza pattern-matching engine
│   ├── neural_net/         # Perceptrons & NumPy MLP for XOR
│   ├── tokenizer/          # Subword Byte Pair Encoding (BPE)
│   ├── embeddings/         # Skip-gram Word2Vec model
│   ├── transformer/        # Causal attention & transformer decoder blocks
│   └── generation/         # Temperature & Top-P samplers
├── tests/
│   ├── test_eliza.py
│   └── test_xor.py
├── pyproject.toml
├── README.md
└── ROADMAP.md

---

## 📌 Architecture Overview

[ Input Text ]
│
▼
[ BPE Tokenizer ]       ─► Maps raw text to subword token IDs
│
▼
[ Token & Pos Embeddings ] ─► Converts IDs into vectors + position encoding
│
▼
[ Transformer Blocks ]  ─► Multi-Head Attention + Feed-Forward Networks
│
▼
[ Softmax & Sampler ]   ─► Applies Temperature & Top-P for next-token prediction

---

## 🚀 Key Modules

1. **Rule-Based Systems**: Pattern matching & state mechanics (`Eliza`).
2. **Perceptrons**: XOR problem solution via pure NumPy backpropagation.
3. **Tokenization**: Subword Byte Pair Encoding (BPE).
4. **Embeddings**: Skip-gram Word2Vec model with vector arithmetic (`king - man + woman = queen`).
5. **Transformer**: Causal multi-head self-attention, layer normalization, and residual connections.
6. **Sampling**: Autoregressive loop with Temperature and Top-P filtering.

---

## 🛠️ Setup & Installation

```bash
# Clone repository
git clone [https://github.com/sjamal/llm-from-scratch.git](https://github.com/sjamal/llm-from-scratch.git)
cd llm-from-scratch

# Install dependencies
pip install -e .

#OR
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e ".[dev]"

# Run test suite
python -m pytest tests/

# Execute text generation runner
python scripts/generate.py --prompt "hello world" --temp 0.7 --top_p 0.9

```

---

## 📜 License
Distributed under the MIT License. See LICENSE for details.

---

### `ROADMAP.md`
```markdown
# Project Roadmap & Technical Milestones

## Phase 1: Historical & Math Foundations
- [x] **01_rule_based**: Implement Eliza-style regex matching and conversation loops.
- [x] **02_neural_net**: Build a single vs. multi-layer perceptron to solve XOR using pure backpropagation in NumPy.

## Phase 2: Representation & Tokenization
- [x] **03_tokenizer**: Build a Byte-Pair Encoding (BPE) algorithm from scratch to learn subword vocabularies.
- [x] **04_embeddings**: Train Skip-gram Word2Vec vectors to capture semantic space and vector operations.

## Phase 3: Transformer Core
- [x] **05_transformer/attention.py**: Scaled dot-product attention with causal mask ($Q, K, V$).
- [x] **05_transformer/decoder.py**: Multi-head attention block, LayerNorm, and Feed-Forward Neural Network (FFN).

## Phase 4: Inference & Generation
- [x] **06_generation**: Temperature, Top-K, and Top-P (Nucleus) sampling logic.
- [x] **CLI Scripts**: Training and inference scripts for generating stories on small datasets.

---

## Repository Workflow & AI Guidelines

This repository follows a disciplined engineering workflow:
- **Default Branch**: `develop` for active feature integration.
- **Production Branch**: `main` for stable releases.
- **AI Workspace Rules**: Standardized project constraints, code style, and architectural guidelines are defined in `.cursorrules`.
- **Privacy & Security**: Commit history and documentation strictly use non-sensitive anonymous identifiers.




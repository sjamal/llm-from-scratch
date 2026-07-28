import numpy as np
from typing import Dict, List, Tuple

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class SkipGramWord2Vec:
    """Skip-gram Word2Vec embedding model built from scratch in pure NumPy.
    
    Predicts context words given a target center word.
    """

    def __init__(self, embedding_dim: int = 10, window_size: int = 2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.W1: np.ndarray = np.array([])  # Target embeddings (vocab_size x embedding_dim)
        self.W2: np.ndarray = np.array([])  # Context embeddings (embedding_dim x vocab_size)

    def _build_vocab(self, tokens: List[str]) -> None:
        unique_words = sorted(list(set(tokens)))
        self.word2idx = {w: i for i, w in enumerate(unique_words)}
        self.idx2word = {i: w for i, w in enumerate(unique_words)}
        
        vocab_size = len(unique_words)
        # Initialize weight matrices with small random values
        self.W1 = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        self.W2 = np.random.randn(self.embedding_dim, vocab_size) * 0.01

    def _generate_training_data(self, tokens: List[str]) -> List[Tuple[int, int]]:
        training_pairs = []
        for i, target_word in enumerate(tokens):
            target_idx = self.word2idx[target_word]
            start = max(0, i - self.window_size)
            end = min(len(tokens), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = self.word2idx[tokens[j]]
                    training_pairs.append((target_idx, context_idx))
        return training_pairs

    def train(self, corpus: str, epochs: int = 500, lr: float = 0.05) -> float:
        tokens = corpus.lower().split()
        if not tokens:
            return 0.0

        self._build_vocab(tokens)
        pairs = self._generate_training_data(tokens)
        vocab_size = len(self.word2idx)

        loss = 0.0
        for _ in range(epochs):
            loss = 0.0
            for target_idx, context_idx in pairs:
                # 1. Forward Pass
                # One-hot selection equivalent to picking row target_idx
                h = self.W1[target_idx]  # Shape: (embedding_dim,)
                u = np.dot(h, self.W2)    # Shape: (vocab_size,)
                y_pred = softmax(u)      # Shape: (vocab_size,)

                # Cross-entropy loss
                loss -= np.log(y_pred[context_idx] + 1e-10)

                # 2. Backpropagation
                e = y_pred.copy()
                e[context_idx] -= 1.0  # Gradient w.r.t logits (y_pred - y_true)

                # Gradients w.r.t weights
                dW2 = np.outer(h, e)                         # (embedding_dim x vocab_size)
                dW1 = np.dot(self.W2, e)                      # (embedding_dim,)

                # 3. Weight Updates
                self.W2 -= lr * dW2
                self.W1[target_idx] -= lr * dW1

        return float(loss / len(pairs))

    def get_vector(self, word: str) -> np.ndarray:
        """Retrieve vector representation for a word."""
        idx = self.word2idx.get(word.lower())
        if idx is None:
            raise KeyError(f"Word '{word}' not in vocabulary.")
        return self.W1[idx]

    def cosine_similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two word vectors."""
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.dot(v1, v2) / (norm + 1e-10))


if __name__ == "__main__":
    text = "king queen man woman royal monarch prince princess"
    model = SkipGramWord2Vec(embedding_dim=4, window_size=2)
    final_loss = model.train(text, epochs=1000, lr=0.1)
    
    print(f"Final Skip-Gram Loss: {final_loss:.4f}")
    sim = model.cosine_similarity("king", "queen")
    print(f"Cosine Similarity ('king', 'queen'): {sim:.4f}")

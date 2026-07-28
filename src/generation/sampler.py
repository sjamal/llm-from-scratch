import numpy as np
from typing import List, Optional


class AutoregressiveSampler:
    """Logit processors and sampling strategies for autoregressive text generation.
    
    Supports Temperature scaling, Top-K filtering, and Top-P (Nucleus) sampling.
    """

    def __init__(self, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def sample_next_token(self, logits: np.ndarray) -> int:
        """Sample next token index from raw unnormalized logits.

        Args:
            logits: 1D array of token logits of shape (vocab_size,)

        Returns:
            Selected token index (int)
        """
        logits = np.copy(logits)

        # 1. Temperature Scaling
        if self.temperature > 0:
            logits = logits / self.temperature
        else:
            # Greedy choice if temperature is 0
            return int(np.argmax(logits))

        # 2. Top-K Filtering
        if self.top_k > 0 and self.top_k < len(logits):
            indices_to_remove = logits < np.sort(logits)[-self.top_k]
            logits[indices_to_remove] = -np.inf

        # Convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 3. Top-P (Nucleus) Filtering
        if self.top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift indices right to keep first token above threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0
            
            # Re-normalize probabilities
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs = probs / prob_sum

        # 4. Multinominal Random Sampling
        return int(np.random.choice(len(probs), p=probs))


if __name__ == "__main__":
    # Example raw logits for a vocabulary of 5 tokens
    sample_logits = np.array([2.0, 1.5, 0.1, 5.0, 0.8])
    sampler = AutoregressiveSampler(temperature=0.7, top_k=3, top_p=0.9)
    sampled_id = sampler.sample_next_token(sample_logits)
    print(f"Sampled Token ID: {sampled_id}")

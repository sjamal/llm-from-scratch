#!/usr/bin/env python3
"""CLI script to demonstrate end-to-end token generation pipeline."""

import argparse
import numpy as np
from src import BPETokenizer, AutoregressiveSampler


def main():
    parser = argparse.ArgumentParser(description="LLM From Scratch Generation Runner")
    parser.add_argument("--prompt", type=str, default="hello world", help="Input prompt")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature scaling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus threshold")
    parser.add_argument("--steps", type=int, default=5, help="Number of tokens to generate")
    args = parser.parse_args()

    # Train mini-tokenizer for demonstration
    tokenizer = BPETokenizer(vocab_size=25)
    tokenizer.train("hello world hello python llm transformer generation")

    tokens = tokenizer.encode(args.prompt)
    sampler = AutoregressiveSampler(temperature=args.temp, top_p=args.top_p)

    print(f"Prompt: '{args.prompt}'")
    print(f"Encoded Tokens: {tokens}")

    # Simulated autoregressive generation loop
    vocab_size = len(tokenizer.vocab) if tokenizer.vocab else 25
    generated_tokens = list(tokens)

    for step in range(args.steps):
        # Simulated model output logits
        simulated_logits = np.random.randn(vocab_size)
        next_id = sampler.sample_next_token(simulated_logits)
        
        # Pick corresponding subword token from vocabulary if available
        if next_id < len(tokenizer.vocab):
            next_token = tokenizer.vocab[next_id]
        else:
            next_token = "a"
        
        generated_tokens.append(next_token)

    decoded = tokenizer.decode(generated_tokens)
    print(f"Generated Output: {decoded}")


if __name__ == "__main__":
    main()

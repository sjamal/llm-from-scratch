import pytest
import numpy as np
from src import AutoregressiveSampler


def test_greedy_sampling_zero_temp():
    sampler = AutoregressiveSampler(temperature=0.0)
    logits = np.array([1.0, 3.5, 0.2, 2.1])
    # Temperature 0 must always pick max logit (index 1)
    sampled = sampler.sample_next_token(logits)
    assert sampled == 1


def test_top_k_sampling_restriction():
    sampler = AutoregressiveSampler(temperature=1.0, top_k=1)
    logits = np.array([0.1, 10.0, 0.2])
    # Top-K=1 must force selecting highest logit index
    sampled = sampler.sample_next_token(logits)
    assert sampled == 1


def test_top_p_nucleus_sampling():
    sampler = AutoregressiveSampler(temperature=1.0, top_p=0.5)
    logits = np.array([10.0, 0.01, 0.01])
    # Index 0 carries > 99% probability, so nucleus top_p=0.5 must pick index 0
    sampled = sampler.sample_next_token(logits)
    assert sampled == 0

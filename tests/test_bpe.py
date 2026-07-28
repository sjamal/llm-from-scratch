import pytest
from src import BPETokenizer

def test_bpe_training_and_encoding():
    corpus = "low lower newest widest"
    tokenizer = BPETokenizer(vocab_size=15)
    tokenizer.train(corpus)
    
    tokens = tokenizer.encode("lower")
    assert isinstance(tokens, list)
    assert len(tokens) > 0

def test_bpe_decoding():
    tokenizer = BPETokenizer(vocab_size=10)
    tokenizer.train("hello world")
    
    original = "hello"
    tokens = tokenizer.encode(original)
    decoded = tokenizer.decode(tokens)
    
    assert decoded == original

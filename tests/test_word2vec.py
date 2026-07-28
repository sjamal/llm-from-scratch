import pytest
import numpy as np
from src import SkipGramWord2Vec

@pytest.fixture
def trained_model():
    text = "apple banana fruit apple fruit banana"
    model = SkipGramWord2Vec(embedding_dim=4, window_size=1)
    model.train(text, epochs=200, lr=0.1)
    return model

def test_word2vec_get_vector(trained_model):
    vec = trained_model.get_vector("apple")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (4,)

def test_word2vec_similarity(trained_model):
    sim = trained_model.cosine_similarity("apple", "banana")
    assert -1.0 <= sim <= 1.0

def test_word2vec_missing_word(trained_model):
    with pytest.raises(KeyError):
        trained_model.get_vector("unseenword")

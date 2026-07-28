import pytest
import numpy as np
from src import SingleLayerPerceptron, MultiLayerPerceptronXOR

@pytest.fixture
def xor_data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    return X, y

def test_single_layer_perceptron_fails_xor(xor_data):
    X, y = xor_data
    slp = SingleLayerPerceptron()
    loss = slp.train(X, y, epochs=2000)
    # SLP cannot achieve MSE loss near 0 on non-linear XOR problem
    assert loss > 0.1

def test_mlp_solves_xor(xor_data):
    X, y = xor_data
    mlp = MultiLayerPerceptronXOR()
    loss = mlp.train(X, y, epochs=10000, lr=0.5)
    # MLP should converge to near zero MSE loss
    assert loss < 0.02
    
    _, predictions = mlp.forward(X)
    rounded_preds = np.round(predictions)
    np.testing.assert_array_equal(rounded_preds, y)

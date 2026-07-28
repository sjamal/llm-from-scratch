import json
import numpy as np
from typing import Dict, Tuple

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)

class SingleLayerPerceptron:
    """A single-layer perceptron. Demonstrates failure to learn non-linear XOR."""
    
    def __init__(self, input_dim: int = 2):
        self.weights = np.random.randn(input_dim, 1)
        self.bias = np.random.randn(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 5000, lr: float = 0.1) -> float:
        for _ in range(epochs):
            output = self.forward(X)
            error = y - output
            # Gradient descent
            d_weights = np.dot(X.T, error * output * (1 - output))
            d_bias = np.sum(error * output * (1 - output))
            self.weights += lr * d_weights
            self.bias += lr * d_bias
        
        final_loss = float(np.mean((y - self.forward(X)) ** 2))
        return final_loss


class MultiLayerPerceptronXOR:
    """2-layer MLP capable of solving XOR via backpropagation."""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, output_dim: int = 1):
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a1, self.a2

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10000, lr: float = 0.5) -> float:
        for _ in range(epochs):
            a1, a2 = self.forward(X)
            
            # Backpropagation
            error_output = y - a2
            d_output = error_output * (a2 * (1.0 - a2))
            
            error_hidden = np.dot(d_output, self.W2.T)
            d_hidden = error_hidden * (a1 * (1.0 - a1))
            
            # Update weights
            self.W2 += np.dot(a1.T, d_output) * lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
            self.W1 += np.dot(X.T, d_hidden) * lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
            
        final_loss = float(np.mean((y - self.forward(X)[1]) ** 2))
        return final_loss

    def export_weights(self, filepath: str) -> None:
        """Export trained weights into a JSON file format."""
        data = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    slp = SingleLayerPerceptron()
    slp_loss = slp.train(X, y)
    print(f"Single Layer Loss on XOR (Should fail to converge near 0): {slp_loss:.4f}")

    mlp = MultiLayerPerceptronXOR()
    mlp_loss = mlp.train(X, y)
    print(f"Multi-Layer MLP Loss on XOR (Should converge close to 0): {mlp_loss:.4f}")
    
    _, predictions = mlp.forward(X)
    print("Predictions on XOR:")
    for inp, pred in zip(X, predictions):
        print(f"  Input: {inp} -> Predicted: {pred[0]:.4f} (Rounded: {round(pred[0])})")

import numpy as np

from core.model import Model, sigmoid


class NeuralNetwork(Model):
    """One hidden layer (ReLU) + sigmoid output."""

    def __init__(self, n_features, n_hidden=8, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, 0.5, (n_features, n_hidden)),
            "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, 0.5, n_hidden),
            "b2": 0.0,
        }

    def forward(self, X):
        p = self.params
        self.h = np.maximum(0, X @ p["W1"] + p["b1"])  # hidden layer (kept for backprop)
        return sigmoid(self.h @ p["W2"] + p["b2"])

    def gradients(self, X, y):
        p = self.params
        n = len(X)

        # Output layer
        error = self.forward(X) - y
        dW2 = self.h.T @ error / n
        db2 = np.mean(error)

        # Hidden layer (backpropagation)
        dh = np.outer(error, p["W2"]) * (self.h > 0)
        dW1 = X.T @ dh / n
        db1 = dh.mean(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

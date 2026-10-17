import numpy as np

from core.model import Model, sigmoid


class LogisticRegression(Model):
    """p = sigmoid(X.W + b)"""

    def __init__(self, n_features):
        super().__init__()
        self.params = {"W": np.zeros(n_features), "b": 0.0}

    def forward(self, X):
        return sigmoid(X @ self.params["W"] + self.params["b"])

    def gradients(self, X, y):
        n = len(X)
        error = self.forward(X) - y
        return {"W": X.T @ error / n, "b": np.mean(error)}

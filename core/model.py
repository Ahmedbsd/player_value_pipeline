import pickle
from abc import ABC, abstractmethod

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Model(ABC):
    """Base class: every model must implement forward and gradients."""

    def __init__(self):
        self.params = {}  # weights, updated by the optimizer

    @abstractmethod
    def forward(self, X):
        """Return probabilities of class 1."""

    @abstractmethod
    def gradients(self, X, y):
        """Return a dict of gradients (same keys as self.params)."""

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

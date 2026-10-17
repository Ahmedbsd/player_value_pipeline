from utils.metrics import binary_cross_entropy


class Trainer:
    """Train any Model with any optimizer."""

    def __init__(self, model, optimizer, epochs=1000, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.history = []  # loss per epoch

    def fit(self, X, y):
        for epoch in range(1, self.epochs + 1):
            grads = self.model.gradients(X, y)
            self.optimizer.step(self.model, grads)

            loss = binary_cross_entropy(y, self.model.forward(X))
            self.history.append(loss)

            if self.verbose and epoch % (self.epochs // 5) == 0:
                print(f"  epoch {epoch:5d} | loss = {loss:.4f}")
        return self.model

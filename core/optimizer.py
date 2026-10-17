class GradientDescent:
    """Update rule: param = param - lr * gradient"""

    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate

    def step(self, model, grads):
        for name in model.params:
            model.params[name] = model.params[name] - self.lr * grads[name]

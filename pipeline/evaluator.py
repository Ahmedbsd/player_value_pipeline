from utils import metrics


class Evaluator:
    """Compute the test scores of a trained model."""

    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        return {
            "accuracy": metrics.accuracy(y, y_pred),
            "precision": metrics.precision(y, y_pred),
            "recall": metrics.recall(y, y_pred),
            "f1": metrics.f1_score(y, y_pred),
        }

    def report(self, X, y, name="Model"):
        scores = self.evaluate(X, y)
        print(f"\n{name} - test results")
        for k, v in scores.items():
            print(f"  {k:<10}: {v:.3f}")
        return scores

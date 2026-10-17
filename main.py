"""Run the full pipeline: data -> train -> evaluate -> compare."""
from core.dataset import Dataset, FEATURES
from core.logistic_regression import LogisticRegression
from core.neural_network import NeuralNetwork
from core.optimizer import GradientDescent
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator

DATA_PATH = "data/player_data.csv"


def main():
    # 1. Data
    dataset = Dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = dataset.prepare()
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | "
          f"High value threshold: €{dataset.threshold:.0f}m")

    # 2. Models (same pipeline for both = polymorphism)
    n = len(FEATURES)
    models = {
        "Logistic Regression": LogisticRegression(n),
        "Neural Network": NeuralNetwork(n, n_hidden=8),
    }

    # 3. Train + evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        Trainer(model, GradientDescent(learning_rate=0.1), epochs=2000).fit(X_train, y_train)
        results[name] = Evaluator(model).report(X_test, y_test, name)

    # 4. Best model by F1
    best = max(results, key=lambda k: results[k]["f1"])
    print(f"\nBest model: {best} (F1 = {results[best]['f1']:.3f})")


if __name__ == "__main__":
    main()

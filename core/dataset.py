import numpy as np
import pandas as pd

from utils.preprocessing import parse_market_value, clean, StandardScaler

# A few important features only
FEATURES = ["Age", "Min", "Gls", "Ast", "Sh", "Touches", "Passes Completed"]
TARGET = "Market value"


class Dataset:
    """Load the CSV, build the target and split into train / test."""

    def __init__(self, path, test_size=0.2, seed=42):
        self.path = path
        self.test_size = test_size
        self.seed = seed
        self.scaler = StandardScaler()

    def load(self):
        df = pd.read_csv(self.path)
        df = clean(df, FEATURES + [TARGET])

        # Target: 1 = high value player (market value >= median), 0 = otherwise
        values = df[TARGET].apply(parse_market_value)
        self.threshold = values.median()
        y = (values >= self.threshold).astype(int).to_numpy()

        X = df[FEATURES].to_numpy(dtype=float)
        return X, y

    def split(self, X, y):
        # Shuffle then cut
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        cut = int(len(X) * (1 - self.test_size))
        train, test = idx[:cut], idx[cut:]

        # Fit the scaler on train only
        X_train = self.scaler.fit_transform(X[train])
        X_test = self.scaler.transform(X[test])
        return X_train, X_test, y[train], y[test]

    def prepare(self):
        """Load + split in one call."""
        X, y = self.load()
        return self.split(X, y)

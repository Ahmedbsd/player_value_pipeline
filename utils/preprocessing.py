import numpy as np
import pandas as pd


def parse_market_value(value):
    """Convert '€27.00m' -> 27.0 (millions of euros)."""
    return float(str(value).replace("€", "").replace("m", ""))


def clean(df, columns):
    """Keep only the needed columns and drop rows with missing values."""
    return df[columns].dropna().reset_index(drop=True)


class StandardScaler:
    """Scale each feature to mean 0 and std 1."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

"""Placeholder for ML models."""

from pathlib import Path
import pandas as pd


def save_ml_metrics(symbol: str, freq: str, metric: float):
    path = Path("data/processed/ml_metrics.csv")
    header = not path.exists()
    df = pd.DataFrame([[symbol, freq, metric]],
                      columns=["symbol", "freq", "metric"])
    df.to_csv(path, mode="a", header=header, index=False)


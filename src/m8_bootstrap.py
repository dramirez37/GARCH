"""Bootstrap utilities."""

from pathlib import Path
import pandas as pd
import numpy as np

from .utils import bootstrap_block


def bootstrap_periods(series: pd.Series, block: int = 20, reps: int = 100):
    return bootstrap_block(series, block, reps)


def save_bootstrap_results(symbol: str, freq: str, arr: np.ndarray):
    path = Path("data/processed/bootstrap_results.csv")
    header = not path.exists()
    df = pd.DataFrame(arr)
    df.insert(0, "symbol", symbol)
    df.insert(1, "freq", freq)
    df.to_csv(path, mode="a", header=header, index=False)


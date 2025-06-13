"""GARCH-related model fits."""

from pathlib import Path
import pandas as pd
from arch import arch_model

from .utils import load_csv


def load_data(symbol: str, freq: str) -> pd.Series:
    path = Path("data/processed/prices_real") / f"{symbol}_{freq}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].pct_change().dropna()


def fit_garch_baseline(returns: pd.Series):
    model = arch_model(returns * 100, vol="Garch", p=1, q=1).fit(disp="off")
    return model


def append_model_metrics(model_name: str, symbol: str, freq: str, aic: float):
    path = Path("data/processed/model_metrics.csv")
    header = not path.exists()
    df = pd.DataFrame([[model_name, symbol, freq, aic]],
                      columns=["model", "symbol", "freq", "aic"])
    df.to_csv(path, mode="a", header=header, index=False)


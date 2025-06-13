"""Logistic regression combining spread and cycle phase."""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_spread(symbol: str) -> pd.Series:
    path = Path("data/processed/spreads") / f"{symbol}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def load_phase(symbol: str, freq: str) -> pd.Series:
    path = Path("data/processed/resid") / f"{symbol}_{freq}_resid.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return (df.iloc[:, 0] > 0).astype(int)


def fit_logistic(spread: pd.Series, phase: pd.Series):
    common = spread.dropna().index.intersection(phase.index)
    X = spread.loc[common].values.reshape(-1, 1)
    y = phase.loc[common].values
    model = LogisticRegression().fit(X, y)
    return model


def save_macro_models(symbol: str, freq: str, coef: float):
    path = Path("data/processed/macro_models.csv")
    header = not path.exists()
    df = pd.DataFrame([[symbol, freq, coef]],
                      columns=["symbol", "freq", "coef"])
    df.to_csv(path, mode="a", header=header, index=False)


"""Fit simple ARIMA models to returns."""

from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .utils import load_csv


def load_returns(symbol: str, freq: str) -> pd.Series:
    path = Path("data/processed/prices_real") / f"{symbol}_{freq}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0].pct_change().dropna()


def select_arima_order(series: pd.Series):
    # placeholder: always return (1,0,0)
    return (1, 0, 0)


def fit_arima(series: pd.Series) -> ARIMA:
    order = select_arima_order(series)
    model = ARIMA(series, order=order).fit()
    return model


def append_model_metrics(model_name: str, symbol: str, freq: str, aic: float):
    path = Path("data/processed/model_metrics.csv")
    header = not path.exists()
    df = pd.DataFrame([[model_name, symbol, freq, aic]],
                      columns=["model", "symbol", "freq", "aic"])
    df.to_csv(path, mode="a", header=header, index=False)


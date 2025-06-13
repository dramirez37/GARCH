"""Compute returns and rolling volatility, then remove calendar effects."""

import numpy as np
from pathlib import Path
import pandas as pd

from .utils import resample


def compute_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def rolling_vols(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * (window ** 0.5)


def deseasonalize(vol: pd.Series) -> pd.Series:
    dummies = pd.get_dummies(vol.index.month)
    dummies.index = vol.index
    beta = np.linalg.lstsq(dummies, vol.fillna(0), rcond=None)[0]
    seasonal = dummies.dot(beta)
    return vol - seasonal


def save_vols_and_resid(symbol: str, freq: str, vol: pd.Series, resid: pd.Series):
    vol_dir = Path("data/processed/vols")
    res_dir = Path("data/processed/resid")
    vol_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    vol.to_csv(vol_dir / f"{symbol}_{freq}_vol.csv", header=["vol"])
    resid.to_csv(res_dir / f"{symbol}_{freq}_resid.csv", header=["resid"])



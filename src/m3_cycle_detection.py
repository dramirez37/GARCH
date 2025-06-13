"""Detect cycle periods from residual volatility using FFT."""

from pathlib import Path
import numpy as np
import pandas as pd

from .utils import detect_period


def load_resid(symbol: str, freq: str) -> pd.Series:
    path = Path("data/processed/resid") / f"{symbol}_{freq}_resid.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def detect_period_series(series: pd.Series) -> float:
    return detect_period(series.dropna())


def plot_periodogram(series: pd.Series, T: float, path: Path):
    import matplotlib.pyplot as plt
    yf = abs(np.fft.rfft(series - series.mean())) ** 2
    xf = np.fft.rfftfreq(len(series), 1)
    plt.figure()
    plt.plot(xf[1:], yf[1:])
    plt.axvline(1 / T, color="r", linestyle="--")
    plt.xlabel("Frequency")
    plt.title("Periodogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def fit_cycle(series: pd.Series, T: float) -> pd.Series:
    idx = series.dropna().index
    phase = 2 * np.pi * (idx - idx[0]).days / T
    fit = np.sin(phase)
    return pd.Series(fit, index=idx)


def record_and_save(symbol: str, freq: str, T: float):
    df = pd.DataFrame([[symbol, freq, T]], columns=["symbol", "freq", "period"])
    path = Path("data/processed/cycle_periods.csv")
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


"""Utility functions for data loading, resampling, and cycle detection."""

from pathlib import Path
from typing import Dict, Iterable
import pandas as pd
import numpy as np


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV with a two-row header and date column."""
    header = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.lower().startswith("date"):
                header = i
                break
    return pd.read_csv(path, skiprows=header, on_bad_lines="skip")


def resample(series: pd.Series, freq: str, start_date: str = None) -> pd.Series:
    """Resample a price series to the given frequency using last observations."""
    s = series.copy()
    if start_date:
        s = s[s.index >= pd.to_datetime(start_date)]
    if freq == "D":
        return s.asfreq("D").ffill()
    rule = {"W": "W-FRI", "M": "M"}.get(freq, freq)
    return s.resample(rule).last().dropna()


def detect_period(series: pd.Series) -> float:
    """Return the dominant cycle period via FFT."""
    vals = series.dropna().values
    if len(vals) < 4:
        return float("nan")
    n = len(vals)
    yf = np.fft.rfft(vals - vals.mean())
    xf = np.fft.rfftfreq(n, 1)
    idx = np.abs(yf)[1:].argmax() + 1
    return float(1 / xf[idx])


def bootstrap_block(series: Iterable[float], block: int, reps: int = 100) -> np.ndarray:
    """Simple moving block bootstrap."""
    arr = np.array(list(series))
    n = len(arr)
    if n == 0:
        return np.empty((reps, 0))
    indices = np.arange(n)
    out = np.empty((reps, n))
    for r in range(reps):
        samples = []
        while len(samples) < n:
            start = np.random.randint(0, n - block + 1)
            samples.extend(indices[start:start+block])
        out[r] = arr[samples[:n]]
    return out


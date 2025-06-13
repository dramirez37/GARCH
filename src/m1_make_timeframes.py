"""Convert mixed-frequency raw prices into clean monthly/weekly/daily files."""

from pathlib import Path
import pandas as pd

from .utils import load_csv, resample


def load_raw(symbol: str) -> pd.Series:
    """Load raw CSV for the given symbol and return the real close series."""
    path = Path("mainz") / f"{symbol}.csv"
    df = load_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date')
    col = [c for c in df.columns if c.lower().endswith('real_close')]
    if not col:
        col = [c for c in df.columns if c.lower().endswith('close')]
    if not col:
        raise ValueError('No close column found')
    return pd.to_numeric(df[col[0]], errors='coerce').dropna()


def detect_switch_points(df: pd.Series):
    """Placeholder for frequency switch detection."""
    return df.index[0]


def resample_to(series: pd.Series, freq: str, start_date) -> pd.Series:
    return resample(series, freq, start_date)


def save_prices_real(symbol: str, freq: str, series: pd.Series):
    out_dir = Path("data/processed/prices_real")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{freq}.csv"
    series.to_csv(path, header=["Close"])


def main(symbol: str):
    raw = load_raw(symbol)
    start = detect_switch_points(raw)
    for freq in ("M", "W", "D"):
        ser = resample_to(raw, freq, start)
        save_prices_real(symbol, freq, ser)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.1_make_timeframes SYMBOL")
        sys.exit(1)
    main(sys.argv[1])

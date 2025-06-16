"""Triptych GARCH regime labeling engine."""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from arch.univariate import GARCH, ConstantMean, Normal


# ---------------------------------------------------------------------------
# Module 1: Data Ingestion and Timeframe Aggregation
# ---------------------------------------------------------------------------

def load_prices(path: Path) -> pd.DataFrame:
    """Return daily OHLCV data with a ``DatetimeIndex``.

    The data files shipped with ``mainz.zip`` include two metadata rows before
    the actual header. ``pandas.read_csv`` would otherwise treat the first line
    as the header and fail to parse the date column.  We therefore skip those two
    lines and explicitly parse dates using the ``mm/dd/YYYY`` format used in the
    files.
    """

    df = pd.read_csv(
        path,
        skiprows=2,  # drop metadata rows
        parse_dates=["Date"],
        date_parser=lambda s: pd.to_datetime(s, format="%m/%d/%Y"),
        index_col="Date",
    )

    df = df.sort_index()
    df = df.ffill().dropna()
    return df


def calc_log_returns(close: pd.Series) -> pd.Series:
    """Compute daily logarithmic returns."""
    return np.log(close).diff().dropna()


def resample_returns(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (monthly, weekly, daily) log return series."""
    daily_r = calc_log_returns(close)
    weekly = close.resample("W-FRI").last()
    monthly = close.resample("M").last()
    weekly_r = calc_log_returns(weekly)
    monthly_r = calc_log_returns(monthly)
    return monthly_r, weekly_r, daily_r


# ---------------------------------------------------------------------------
# Module 2: GARCH Model Fitting Engine
# ---------------------------------------------------------------------------

def fit_gjr_garch(returns: pd.Series):
    """Fit AR(1)-GJR-GARCH(1,1) model."""
    am = ConstantMean(returns * 100)
    am.volatility = GARCH(p=1, o=1, q=1)
    am.distribution = Normal()
    res = am.fit(disp="off")
    return res


def conditional_variance(model) -> pd.Series:
    """Return one-step-ahead conditional variance."""
    var = model.conditional_volatility ** 2 / 10000  # back to return units
    var.index = model.model._y.index
    return var


# ---------------------------------------------------------------------------
# Module 3: Volatility State Classifier
# ---------------------------------------------------------------------------

def dynamic_threshold(var: pd.Series, window: int) -> pd.Series:
    """Rolling mean as adaptive threshold."""
    return var.rolling(window, min_periods=1).mean()


def assign_states(var: pd.Series, thresh: pd.Series) -> pd.Series:
    """Binary state assignment."""
    return np.where(var > thresh, "HV", "LV")


# ---------------------------------------------------------------------------
# Module 4: Triptych Label Synthesizer
# ---------------------------------------------------------------------------

def synchronize_states(dm: pd.Series, dw: pd.Series, dd: pd.Series) -> pd.DataFrame:
    """Align monthly, weekly and daily states to a daily index."""
    idx = dd.index
    df = pd.DataFrame({"S_D": dd}, index=idx)
    df["S_W"] = dw.reindex(idx, method="ffill")
    df["S_M"] = dm.reindex(idx, method="ffill")
    return df


def generate_label(row: pd.Series) -> str:
    mapping = {
        ("LV", "LV", "LV"): "Grand Squeeze",
        ("HV", "LV", "LV"): "Calm Daily",
        ("LV", "HV", "LV"): "Turbulent Week",
        ("LV", "LV", "HV"): "Flare Up",
    }
    key = (row["S_M"], row["S_W"], row["S_D"])
    return mapping.get(key, "Mixed")


def triptych_labels(path: Path, weekly_win: int = 52, monthly_win: int = 12) -> pd.DataFrame:
    """Main entry point. Return DataFrame with states and label."""
    data = load_prices(path)
    monthly_r, weekly_r, daily_r = resample_returns(data["Close"])

    m_model = fit_gjr_garch(monthly_r)
    w_model = fit_gjr_garch(weekly_r)
    d_model = fit_gjr_garch(daily_r)

    m_var = conditional_variance(m_model)
    w_var = conditional_variance(w_model)
    d_var = conditional_variance(d_model)

    m_thresh = dynamic_threshold(m_var, monthly_win)
    w_thresh = dynamic_threshold(w_var, weekly_win)
    d_thresh = dynamic_threshold(d_var, 252)

    m_state = pd.Series(assign_states(m_var, m_thresh), index=m_var.index)
    w_state = pd.Series(assign_states(w_var, w_thresh), index=w_var.index)
    d_state = pd.Series(assign_states(d_var, d_thresh), index=d_var.index)

    df = synchronize_states(m_state, w_state, d_state)
    df["label"] = df.apply(generate_label, axis=1)
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python garch_triptych.py PRICE_CSV")
        sys.exit(1)

    csv = Path(sys.argv[1])
    res = triptych_labels(csv)
    print(res.tail())

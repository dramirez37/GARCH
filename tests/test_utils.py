import numpy as np
import pandas as pd
from src import utils


def test_resample():
    idx = pd.date_range('2020-01-01', periods=10, freq='D')
    series = pd.Series(range(10), index=idx)
    result = utils.resample(series, 'W')
    assert len(result) == 2
    assert result.iloc[0] == series.iloc[2]


def test_detect_period():
    x = pd.Series(np.sin(2 * np.pi * np.arange(20) / 5))
    T = utils.detect_period(x)
    assert abs(T - 5) < 0.5


def test_bootstrap_block():
    arr = utils.bootstrap_block(range(10), block=3, reps=2)
    assert arr.shape == (2, 10)

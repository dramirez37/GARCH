import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, coint
from statsmodels.tsa.statespace.markov_regression import MarkovRegression
from arch import arch_model
from scipy.fft import rfft, rfftfreq
from scipy.signal import coherence
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_curve, auc
import pywt

# 1. Load & Reshape

def find_header_row(csv_path):
    """Return the zero-indexed row number containing the column names."""
    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip().lower().startswith("date"):
                return i
    return 0


def load_and_melt(csv_path):
    header_row = find_header_row(csv_path)
    df = pd.read_csv(csv_path, skiprows=header_row, header=0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = df.columns.str.strip()
    long_df = (
        df.reset_index()
        .melt(id_vars='Date', var_name='Ticker_Field', value_name='Value')
    )
    long_df[['Ticker', 'Field']] = long_df['Ticker_Field'].str.split('_', 1, expand=True)
    return long_df[['Date', 'Ticker', 'Field', 'Value']]

# 2. Select Price Series

def select_close(df_long):
    close_df = df_long[df_long['Field'].isin(['Close', 'AdjClose'])]
    return close_df.dropna()

# 3. Define Sliding Window

def restrict_window(close_df, years=30):
    max_date = close_df['Date'].max()
    start_date = max_date - pd.DateOffset(years=years)
    return close_df[close_df['Date'] >= start_date]

# 4. Native-Resolution Detection

def detect_daily_start(df):
    df = df.sort_values('Date')
    df['Gap'] = df.groupby('Ticker')['Date'].diff()
    def first_good(sub):
        one_year = pd.Timedelta(days=365)
        roll = sub.set_index('Date')['Gap'].fillna(pd.Timedelta(days=1)).gt(pd.Timedelta(days=3))
        coverage = (~roll).rolling('365D').mean()
        mask = coverage >= 0.95
        if mask.any():
            return mask[mask].index[0]
        return sub['Date'].iloc[0]
    return df.groupby('Ticker').apply(first_good).to_dict()

# 5. Derive Timeframes & Returns

def make_timeframes(df, start_map):
    out = {}
    for ticker, start in start_map.items():
        sub = df[(df['Ticker']==ticker) & (df['Date']>=start)].sort_values('Date')
        sub = sub.set_index('Date')['Value']
        daily = sub
        weekly = sub.resample('W-FRI').last()
        monthly = sub.resample('M').last()
        out[ticker] = {
            'D': np.log(daily).diff().dropna(),
            'W': np.log(weekly).diff().dropna(),
            'M': np.log(monthly).diff().dropna()
        }
    return out

# 6. Realized Volatility & Deseasonalization

def realized_vol(series, window):
    return series.rolling(window).std() * np.sqrt(window)

from pandas.api.types import CategoricalDtype

def deseasonalize(vol, freq):
    if freq == 'D':
        cat = vol.index.dayofweek
    elif freq == 'W':
        cat = vol.index.month
    else:
        cat = vol.index.month
    dummies = pd.get_dummies(cat)
    beta = np.linalg.lstsq(dummies, vol.fillna(0), rcond=None)[0]
    seasonal = dummies.dot(beta)
    return vol - seasonal

# 7. Cycle Detection

def detect_cycle(vol):
    vol = vol.dropna()
    N = len(vol)
    yf = rfft(vol - vol.mean())
    xf = rfftfreq(N, 1)
    power = np.abs(yf)**2
    idx = power[1:].argmax()+1
    period = 1/xf[idx]
    return period


# Additional Analysis Functions

def rolling_correlations(vol_dict, window=252):
    """Return a DataFrame of rolling correlations for each pair of tickers."""
    df = pd.DataFrame(vol_dict)
    return df.rolling(window).corr().dropna()


def cointegration_matrix(price_df):
    """Compute p-values of cointegration tests between all pairs."""
    tickers = price_df.columns
    pvals = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i+1:]:
            _, p, _ = coint(price_df[t1].dropna(), price_df[t2].dropna())
            pvals.loc[t1, t2] = p
            pvals.loc[t2, t1] = p
    return pvals


def cross_spectral_density(series_a, series_b):
    """Return frequencies and coherence between two volatility series."""
    s1 = series_a.dropna()
    s2 = series_b.dropna()
    common = s1.index.intersection(s2.index)
    f, coh = coherence(s1.loc[common], s2.loc[common], fs=1)
    return f, coh


def pca_volatility(vol_dict, n_components=3):
    """Perform PCA on the aligned volatility series."""
    df = pd.DataFrame(vol_dict).dropna()
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(df)
    return pca, pd.DataFrame(comps, index=df.index)


def cluster_assets(cycle_map, n_clusters=3):
    """Cluster tickers based on detected cycle periods."""
    tickers = list(cycle_map.keys())
    periods = np.array([list(cycle_map[t].values()) for t in tickers])
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(periods)
    return pd.Series(labels, index=tickers)

if __name__ == '__main__':
    csv_name = 'David_Ramirez_Mainv2_20250612134107.csv'
    if not Path(csv_name).exists():
        with zipfile.ZipFile('mainz.zip') as z:
            target = [n for n in z.namelist() if n.endswith('.csv')][0]
            with z.open(target) as f:
                Path(csv_name).write_bytes(f.read())
    df_long = load_and_melt(csv_name)
    close_df = select_close(df_long)
    close_df = restrict_window(close_df)
    start_map = detect_daily_start(close_df)
    tf_returns = make_timeframes(close_df, start_map)

    results = {}
    for ticker, maps in tf_returns.items():
        results[ticker] = {}
        for freq, ret in maps.items():
            vol = realized_vol(ret, 10)
            vol = deseasonalize(vol, freq)
            T = detect_cycle(vol)
            results[ticker][freq] = T
            print(f'{ticker} {freq} cycle period: {T:.2f}')

    # Cross-asset analyses using daily volatility
    daily_vol = {t: realized_vol(r['D'], 10) for t, r in tf_returns.items()}
    corr_df = rolling_correlations(daily_vol)
    corr_df.to_csv('rolling_correlations.csv')

    close_wide = close_df.pivot(index='Date', columns='Ticker', values='Value')
    coint_pvals = cointegration_matrix(close_wide)
    coint_pvals.to_csv('cointegration_pvalues.csv')

    pca, pca_series = pca_volatility(daily_vol)
    pca_series.to_csv('volatility_pca.csv')

    cluster_labels = cluster_assets(results)
    cluster_labels.to_csv('cycle_clusters.csv')

    # Further modelling and reporting would be implemented here

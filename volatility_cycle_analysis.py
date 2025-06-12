import zipfile
from pathlib import Path
import io
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, coint
from arch import arch_model
from scipy.fft import rfft, rfftfreq
from scipy.signal import coherence
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_curve, auc
import pywt

plt.style.use("seaborn-v0_8")

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
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    df.columns = df.columns.str.strip()
    long_df = (
        df.reset_index()
        .melt(id_vars='Date', var_name='Ticker_Field', value_name='Value')
    )
    long_df[['Ticker', 'Field']] = long_df['Ticker_Field'].str.split('_', n=1, expand=True)
    return long_df[['Date', 'Ticker', 'Field', 'Value']]


def load_zip_long(zip_path):
    """Return long-format DataFrame and metadata from all CSVs within a zip."""
    frames = []
    meta_records = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith('.csv'):
                continue
            with z.open(name) as f:
                lines = f.read().decode('utf-8').splitlines()

            meta = {}
            if len(lines) >= 2 and 'Series Type' in lines[0]:
                try:
                    reader = csv.DictReader(lines[:2])
                    meta = next(reader)
                except Exception:
                    meta = {}

            header = 0
            for i, line in enumerate(lines):
                if line.lower().startswith('date'):
                    header = i
                    break

            data_lines = []
            for line in lines[header:]:
                low = line.lower()
                if low.startswith('id,') or low.startswith('quota'):
                    break
                data_lines.append(line)

            if not data_lines:
                continue
            df = pd.read_csv(io.StringIO('\n'.join(data_lines)))
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df.columns = df.columns.str.strip()
            long = df.melt(id_vars='Date', var_name='Ticker_Field', value_name='Value')
            long[['Ticker', 'Field']] = long['Ticker_Field'].str.split('_', n=1, expand=True)
            frames.append(long[['Date', 'Ticker', 'Field', 'Value']])

            if meta:
                meta_records.append({'Ticker': meta.get('Ticker', ''),
                                     'Series Type': meta.get('Series Type', '').strip()})

    if not frames:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Field', 'Value']), pd.DataFrame()

    df_long = pd.concat(frames, ignore_index=True)
    meta_df = pd.DataFrame(meta_records).drop_duplicates('Ticker')
    return df_long, meta_df

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
    dummies.index = vol.index  # align for arithmetic
    beta = np.linalg.lstsq(dummies, vol.fillna(0), rcond=None)[0]
    seasonal = dummies.dot(beta)
    return vol - seasonal

# 7. Cycle Detection

def detect_cycle(vol):
    vol = vol.dropna()
    if len(vol) < 4:
        return np.nan
    N = len(vol)
    yf = rfft(vol - vol.mean())
    xf = rfftfreq(N, 1)
    power = np.abs(yf) ** 2
    idx = power[1:].argmax() + 1
    period = 1 / xf[idx]
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
            try:
                _, p, _ = coint(price_df[t1].dropna(), price_df[t2].dropna())
            except Exception:
                p = np.nan
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


def coherence_matrix(vol_dict):
    """Return matrix of average coherence between each pair of tickers."""
    tickers = list(vol_dict.keys())
    mat = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    for i, t1 in enumerate(tickers):
        mat.loc[t1, t1] = 1.0
        for t2 in tickers[i + 1:]:
            _, coh = cross_spectral_density(vol_dict[t1], vol_dict[t2])
            avg = float(np.nanmean(coh)) if len(coh) else np.nan
            mat.loc[t1, t2] = avg
            mat.loc[t2, t1] = avg
    return mat


def pca_volatility(vol_dict, n_components=3):
    """Perform PCA on the aligned volatility series."""
    df = pd.DataFrame(vol_dict).dropna()
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(df)
    return pca, pd.DataFrame(comps, index=df.index)


def cluster_assets(cycle_map, n_clusters=3):
    """Cluster tickers based on detected cycle periods."""
    tickers = []
    rows = []
    for t, d in cycle_map.items():
        vals = list(d.values())
        if any(np.isnan(vals)):
            continue
        tickers.append(t)
        rows.append(vals)
    if not rows:
        return pd.Series(dtype=int)
    periods = np.array(rows)
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(periods)
    return pd.Series(labels, index=tickers)

def wavelet_coherence(series_a, series_b, wavelet="morl", scales=None):
    """Compute average wavelet coherence between two series."""
    s1 = series_a.dropna()
    s2 = series_b.dropna()
    common = s1.index.intersection(s2.index)
    if common.empty:
        return np.array([])
    a = s1.loc[common].values
    b = s2.loc[common].values
    if scales is None:
        N = len(common)
        scales = np.arange(1, min(128, N // 2))
    c1, _ = pywt.cwt(a, scales, wavelet)
    c2, _ = pywt.cwt(b, scales, wavelet)
    S1 = np.abs(c1) ** 2
    S2 = np.abs(c2) ** 2
    X = c1 * np.conj(c2)
    S12 = np.abs(X) ** 2
    def smooth(mat):
        kernel = np.ones(3) / 3.0
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 1, mat)
    WCOH = smooth(S12) / (smooth(S1) * smooth(S2))
    return WCOH.mean(axis=1)

def wavelet_coherence_matrix(vol_dict):
    """Return matrix of average wavelet coherence between pairs."""
    tickers = list(vol_dict.keys())
    mat = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
    for i, t1 in enumerate(tickers):
        mat.loc[t1, t1] = 1.0
        for t2 in tickers[i + 1:]:
            coh = wavelet_coherence(vol_dict[t1], vol_dict[t2])
            avg = float(np.nanmean(coh)) if len(coh) else np.nan
            mat.loc[t1, t2] = avg
            mat.loc[t2, t1] = avg
    return mat

def build_group_map(meta_df):
    groups = {}
    for _, row in meta_df.iterrows():
        grp = row.get("Series Type", "Unknown") or "Unknown"
        groups.setdefault(grp, []).append(row["Ticker"])
    return groups

def group_start_dates(close_df, groups):
    individual = detect_daily_start(close_df)
    result = {}
    for g, ticks in groups.items():
        starts = [individual.get(t) for t in ticks if t in individual]
        if not starts:
            continue
        group_start = max(starts)
        for t in ticks:
            result[t] = group_start
    return result

def analyze_group(name, tickers, tf_returns):
    daily_vol = {t: realized_vol(tf_returns[t]["D"], 10) for t in tickers if t in tf_returns}
    if len(daily_vol) < 2:
        return
    corr_df = rolling_correlations(daily_vol)
    corr_df.to_csv(f"{name}_rolling_correlations.csv")

    price_df = {t: tf_returns[t]["D"].cumsum() for t in tickers if t in tf_returns}
    close_wide = pd.DataFrame(price_df)
    coin_mat = cointegration_matrix(close_wide)
    coin_mat.to_csv(f"{name}_cointegration_pvalues.csv")

    df = pd.DataFrame(daily_vol).dropna()
    if df.shape[0] > 1 and df.shape[1] > 1:
        n_comp = min(3, df.shape[0], df.shape[1])
        pca, comps = pca_volatility(daily_vol, n_components=n_comp)
        comps.to_csv(f"{name}_volatility_pca.csv")

    coh = coherence_matrix(daily_vol)
    coh.to_csv(f"{name}_coherence_matrix.csv")

    wcoh = wavelet_coherence_matrix(daily_vol)
    wcoh.to_csv(f"{name}_wavelet_coherence.csv")


# Visualization and summary helpers

def save_cycle_summary(cycle_map, path="cycle_summary.csv"):
    """Save detected cycle periods for each ticker/frequency."""
    df = (
        pd.DataFrame(cycle_map)
        .T
        .rename_axis("Ticker")
    )
    df.to_csv(path)
    return df


def plot_heatmap(data, title, path):
    """Save heatmap of a DataFrame to disk."""
    if data.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pca_variance(pca, path="pca_scree.png"):
    """Plot cumulative explained variance from a PCA object."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    plt.figure()
    plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_cluster_bars(labels, path="cluster_assignments.png"):
    """Bar plot of cluster assignments for each ticker."""
    plt.figure(figsize=(8, 4))
    labels.sort_index().plot(kind="bar")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if __name__ == '__main__':
    if not Path('mainz.zip').exists():
        raise FileNotFoundError('mainz.zip not found')

    df_long, meta_df = load_zip_long('mainz.zip')
    close_df = select_close(df_long)
    group_map = build_group_map(meta_df)
    start_map = group_start_dates(close_df, group_map)
    tf_returns = make_timeframes(close_df, start_map)

    results = {}
    for ticker, maps in tf_returns.items():
        results[ticker] = {}
        for freq, ret in maps.items():
            vol = realized_vol(ret, 10)
            vol = deseasonalize(vol, freq)
            T = detect_cycle(vol)
            results[ticker][freq] = T
            if np.isnan(T):
                print(f'{ticker} {freq} cycle period: NA')
            else:
                print(f'{ticker} {freq} cycle period: {T:.2f}')

    # Cross-asset analyses within each series type
    for grp, tickers in group_map.items():
        analyze_group(grp.replace(' ', '_'), tickers, tf_returns)

    # Overall cross-asset analysis
    daily_vol = {t: realized_vol(r['D'], 10) for t, r in tf_returns.items()}
    corr_df = rolling_correlations(daily_vol)
    corr_df.to_csv('rolling_correlations.csv')

    close_wide = close_df.pivot(index='Date', columns='Ticker', values='Value')
    coint_pvals = cointegration_matrix(close_wide)
    coint_pvals.to_csv('cointegration_pvalues.csv')

    pca, pca_series = pca_volatility(daily_vol)
    pca_series.to_csv('volatility_pca.csv')

    coh_mat = coherence_matrix(daily_vol)
    coh_mat.to_csv('coherence_matrix.csv')

    cluster_labels = cluster_assets(results)
    cluster_labels.to_csv('cycle_clusters.csv')

    # Generate figures and summary files
    save_cycle_summary(results)
    plot_heatmap(corr_df.groupby(level=0).last(), "Rolling Correlations", "correlation_heatmap.png")
    plot_heatmap(coint_pvals, "Cointegration p-values", "cointegration_heatmap.png")
    plot_heatmap(coh_mat, "Spectral Coherence", "coherence_heatmap.png")
    plot_pca_variance(pca)
    plot_cluster_bars(cluster_labels)

    # Further modelling and reporting would be implemented here

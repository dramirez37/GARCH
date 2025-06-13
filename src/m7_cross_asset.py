"""Basic cross-asset correlation and PCA."""

from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA


def group_by_realm(symbols):
    return {"all": symbols}


def rolling_correlations(data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    return data.rolling(window).corr().dropna()


def perform_pca(data: pd.DataFrame, n_components: int = 3):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(data.dropna())
    return pca, comps


def save_cross_asset_stats(df: pd.DataFrame):
    path = Path("data/processed/cross_asset_stats.csv")
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


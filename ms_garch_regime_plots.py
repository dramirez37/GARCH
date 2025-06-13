import io
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from scipy.signal import hilbert
from statsmodels.discrete.discrete_model import Logit
from ms_garch_full import load_yield_spread

plt.style.use("seaborn-v0_8")


def load_real_close(zip_path: str):
    """Load real closes and ticker countries from a zip archive."""
    frames = []
    countries = {}

    def iso3(name: str) -> str:
        mapping = {
            "United States": "USA",
            "United Kingdom": "GBR",
            "Germany": "DEU",
            "France": "FRA",
            "Japan": "JPN",
            "Korea, Republic of": "KOR",
            "India": "IND",
            "Brazil": "BRA",
            "Australia": "AUS",
            "Mexico": "MEX",
            "Russian Federation": "RUS",
            "China": "CHN",
            "Spain": "ESP",
        }
        return mapping.get(name.strip(), name[:3].upper())

    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
            header = 0
            for i, line in enumerate(lines):
                if line.lower().startswith("date"):
                    header = i
                    break
            if header == 0:
                continue
            meta = list(csv.reader([lines[header - 1]]))[0]
            country = iso3(meta[4]) if len(meta) > 4 else None
            df = pd.read_csv(io.StringIO("\n".join(lines[header:])), engine="python", on_bad_lines="skip")
            real_cols = [c for c in df.columns if c.lower().endswith("real_close")]
            if not real_cols:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df[real_cols[0]] = pd.to_numeric(df[real_cols[0]], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")
            ticker = name.split(".")[0]
            frames.append(df[[real_cols[0]]].rename(columns={real_cols[0]: ticker}))
            if country:
                countries[ticker] = country

    if not frames:
        return pd.DataFrame(), {}

    return pd.concat(frames, axis=1).sort_index(), countries


def realized_vol(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window).std() * np.sqrt(window)


def fit_hmm(series: pd.Series, n_states: int = 2):
    """Fit a simple Gaussian HMM to a return series."""
    arr = series.values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200)
    model.fit(arr)
    states = model.predict(arr)
    prob = model.predict_proba(arr)
    return states, prob, model


def compute_cycle_phase(series: pd.Series) -> pd.Series:
    """Return cycle phase via Hilbert transform."""
    x = series.fillna(method="ffill").values
    analytic = hilbert(x - np.mean(x))
    phase = np.angle(analytic)
    return pd.Series(phase, index=series.index)


def logistic_transition(spread, phase, states):
    """Estimate logistic regression for P(s_{t+1}=H | s_t=L)."""
    s = pd.Series(states, index=spread.index)
    target = ((s.shift(-1) == 1) & (s == 0)).astype(int)
    df = pd.DataFrame({"target": target, "spread": spread, "phase": np.cos(phase)})
    df = df.dropna()
    if df.empty:
        return None
    model = Logit(df["target"], df[["spread", "phase"]]).fit(disp=False)
    return model


def regime_durations(states, state_val=0):
    lens = []
    cur = 0
    for s in states:
        if s == state_val:
            cur += 1
        elif cur:
            lens.append(cur)
            cur = 0
    if cur:
        lens.append(cur)
    return lens


def plot_price_with_regime(price, states, path):
    plt.figure(figsize=(10, 4))
    plt.plot(price.index, price.values, label="Real Close")
    state_series = pd.Series(states, index=price.index)
    for val, color in [(0, "lightblue"), (1, "lightcoral")]:
        mask = state_series == val
        plt.fill_between(price.index, price.min(), price.max(), where=mask, color=color, alpha=0.3)
    plt.title("Price with Regime Shading")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_regime_prob(prob, dates, path):
    plt.figure(figsize=(10, 3))
    plt.plot(dates, prob[:, 1], label="Pr(High Vol)")
    plt.ylim(0, 1)
    plt.title("Regime Probability")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_duration_hist(durations_low, durations_high, path):
    plt.figure(figsize=(6, 4))
    sns.histplot(durations_low, color="blue", label="Low", kde=False, stat="density", alpha=0.5)
    sns.histplot(durations_high, color="red", label="High", kde=False, stat="density", alpha=0.5)
    plt.legend()
    plt.xlabel("Duration")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hazard_survival(durations, state_name, path_prefix):
    durations = np.asarray(durations)
    if len(durations) == 0:
        return
    K = durations.max()
    at_least = np.array([(durations >= k).sum() for k in range(1, K+1)])
    endings = np.array([(durations == k).sum() for k in range(1, K+1)])
    survival = at_least / len(durations)
    hazard = np.divide(endings, at_least, out=np.zeros_like(endings, dtype=float), where=at_least>0)
    plt.figure()
    plt.step(range(1, K+1), survival, where="mid")
    plt.ylabel("S(k)")
    plt.xlabel("k")
    plt.title(f"Survival of {state_name} regime")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_survival.png")
    plt.close()

    plt.figure()
    plt.step(range(1, K+1), hazard, where="mid")
    plt.ylabel("h(k)")
    plt.xlabel("k")
    plt.title(f"Hazard of {state_name} regime")
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_hazard.png")
    plt.close()


def main(zip_path="mainz.zip"):
    if not Path(zip_path).exists():
        raise FileNotFoundError(zip_path)
    prices, meta = load_real_close(zip_path)
    spreads = load_yield_spread(zip_path)
    if prices.empty:
        raise ValueError("No real-close columns found")
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        returns = np.log(series).diff().dropna()
        states, prob, model = fit_hmm(returns)
        vol = realized_vol(returns)
        phase = compute_cycle_phase(vol)
        transition = None
        country = meta.get(ticker)
        if country and not spreads.empty and country in spreads.columns:
            sp = spreads[country].reindex(returns.index, method="ffill")
            transition = logistic_transition(sp, phase, states)
        plot_price_with_regime(series.loc[returns.index], states, f"{ticker}_price_regime.png")
        plot_regime_prob(prob, returns.index, f"{ticker}_regime_prob.png")
        dur_low = regime_durations(states, 0)
        dur_high = regime_durations(states, 1)
        plot_duration_hist(dur_low, dur_high, f"{ticker}_duration_hist.png")
        plot_hazard_survival(dur_low, "Low", f"{ticker}_low")
        plot_hazard_survival(dur_high, "High", f"{ticker}_high")
        if transition is not None:
            sp = spreads[country].reindex(returns.index, method="ffill") if country in spreads.columns else None
            if sp is None or sp.isna().all():
                continue
            grid_x = np.linspace(np.nanmin(sp), np.nanmax(sp), 50)
            grid_y = np.linspace(-1, 1, 50)
            X, Y = np.meshgrid(grid_x, grid_y)
            logits = transition.params["spread"] * X + transition.params["phase"] * Y + transition.params.get("const", 0)
            Z = 1/(1+np.exp(-logits))
            plt.figure(figsize=(6,4))
            cs = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
            plt.colorbar(cs)
            plt.xlabel("Spread")
            plt.ylabel("cos(Phase)")
            plt.title("Transition Probability")
            plt.tight_layout()
            plt.savefig(f"{ticker}_transition_surface.png")
            plt.close()

if __name__ == "__main__":
    main()

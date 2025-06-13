import io
import csv
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_curve, auc

plt.style.use("seaborn-v0_8")


def load_price_series(zip_path: str, real=True):
    """Return DataFrame of real or nominal closes indexed by Date."""
    frames = []
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
            df = pd.read_csv(io.StringIO("\n".join(lines[header:])), engine="python", on_bad_lines="skip")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.set_index("Date")
            cols = [c for c in df.columns if c.lower().endswith("real_close")]
            if not real:
                tmp = [c for c in df.columns if c.lower().endswith("close") and not c.lower().endswith("real_close")]
                cols = tmp
            if not cols:
                continue
            ticker = name.split(".")[0]
            series = pd.to_numeric(df[cols[0]], errors="coerce")
            frames.append(series.rename(ticker))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def load_yield_spread(zip_path: str):
    """Return DataFrame of 10y minus 3m yields for each country."""
    ten = {}
    three = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".csv"):
                continue
            code = name.split(".")[0]
            if len(code) < 5:
                continue
            if code.startswith("IG") and code.endswith("10D"):
                country = code[2:-3]
                df = pd.read_csv(
                    io.StringIO(z.read(name).decode("utf-8", errors="ignore")),
                    skiprows=2,
                    on_bad_lines="skip",
                )
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date")
                ten[country] = pd.to_numeric(df[df.columns[1]], errors="coerce")
            if code.startswith("IT") and code.endswith("3D"):
                country = code[2:-2]
                df = pd.read_csv(
                    io.StringIO(z.read(name).decode("utf-8", errors="ignore")),
                    skiprows=2,
                    on_bad_lines="skip",
                )
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.dropna(subset=["Date"]).set_index("Date")
                three[country] = pd.to_numeric(df[df.columns[1]], errors="coerce")
    spreads = []
    for c in ten:
        if c in three:
            df = pd.DataFrame({"10y": ten[c], "3m": three[c]})
            spreads.append((c, df["10y"] - df["3m"]))
    if not spreads:
        return pd.DataFrame()
    return pd.concat({c: s for c, s in spreads}, axis=1)


def ms_garch_neglog(params, r):
    mu1, mu2, log_o1, log_o2, a1, a2, b1, b2, p00_, p11_ = params
    omega1 = np.exp(log_o1)
    omega2 = np.exp(log_o2)
    alpha1 = 1 / (1 + np.exp(-a1))
    alpha2 = 1 / (1 + np.exp(-a2))
    beta1 = (1 / (1 + np.exp(-b1))) * (1 - alpha1) * 0.99
    beta2 = (1 / (1 + np.exp(-b2))) * (1 - alpha2) * 0.99
    p00 = 1 / (1 + np.exp(-p00_))
    p11 = 1 / (1 + np.exp(-p11_))
    trans = np.array([[p00, 1 - p00], [1 - p11, p11]])
    T = len(r)
    h = np.zeros((T, 2))
    prob = np.zeros((T, 2))
    h[0, 0] = omega1 / (1 - alpha1 - beta1 + 1e-6)
    h[0, 1] = omega2 / (1 - alpha2 - beta2 + 1e-6)
    prob[0] = [0.5, 0.5]
    ll = 0.0
    for t in range(T):
        if t > 0:
            h[t, 0] = omega1 + alpha1 * (r[t - 1] - mu1) ** 2 + beta1 * h[t - 1, 0]
            h[t, 1] = omega2 + alpha2 * (r[t - 1] - mu2) ** 2 + beta2 * h[t - 1, 1]
        dens0 = stats.norm.pdf(r[t], loc=mu1, scale=np.sqrt(h[t, 0] + 1e-12))
        dens1 = stats.norm.pdf(r[t], loc=mu2, scale=np.sqrt(h[t, 1] + 1e-12))
        if t > 0:
            pred = prob[t - 1] @ trans
        else:
            pred = prob[0]
        w = pred * np.array([dens0, dens1])
        denom = w.sum()
        if denom <= 0 or np.isnan(denom):
            return np.inf
        prob[t] = w / denom
        ll += np.log(denom + 1e-12)
    return -ll


def ms_garch_filter(params, r):
    mu1, mu2, log_o1, log_o2, a1, a2, b1, b2, p00_, p11_ = params
    omega1 = np.exp(log_o1)
    omega2 = np.exp(log_o2)
    alpha1 = 1 / (1 + np.exp(-a1))
    alpha2 = 1 / (1 + np.exp(-a2))
    beta1 = (1 / (1 + np.exp(-b1))) * (1 - alpha1) * 0.99
    beta2 = (1 / (1 + np.exp(-b2))) * (1 - alpha2) * 0.99
    p00 = 1 / (1 + np.exp(-p00_))
    p11 = 1 / (1 + np.exp(-p11_))
    trans = np.array([[p00, 1 - p00], [1 - p11, p11]])
    T = len(r)
    h = np.zeros((T, 2))
    prob = np.zeros((T, 2))
    h[0, 0] = omega1 / (1 - alpha1 - beta1 + 1e-6)
    h[0, 1] = omega2 / (1 - alpha2 - beta2 + 1e-6)
    prob[0] = [0.5, 0.5]
    for t in range(T):
        if t > 0:
            h[t, 0] = omega1 + alpha1 * (r[t - 1] - mu1) ** 2 + beta1 * h[t - 1, 0]
            h[t, 1] = omega2 + alpha2 * (r[t - 1] - mu2) ** 2 + beta2 * h[t - 1, 1]
        dens0 = stats.norm.pdf(r[t], loc=mu1, scale=np.sqrt(h[t, 0] + 1e-12))
        dens1 = stats.norm.pdf(r[t], loc=mu2, scale=np.sqrt(h[t, 1] + 1e-12))
        if t > 0:
            pred = prob[t - 1] @ trans
        else:
            pred = prob[0]
        w = pred * np.array([dens0, dens1])
        denom = w.sum()
        prob[t] = w / denom
    return prob, h


def fit_ms_garch(returns):
    returns = returns.dropna().values
    init = np.array([0.0, 0.0, np.log(returns.var()), np.log(returns.var()), 0.2, 0.2, 0.7, 0.7, 0.8, 0.8])
    res = minimize(ms_garch_neglog, init, args=(returns,), method="L-BFGS-B")
    prob, h = ms_garch_filter(res.x, returns)
    states = prob.argmax(axis=1)
    return res, prob, states


def realized_vol(series, window=20):
    return series.rolling(window).std() * np.sqrt(window)


def cycle_phase(series):
    x = series.fillna(method="ffill").values
    analytic = stats.hilbert(x - np.mean(x))
    phase = np.angle(analytic)
    return pd.Series(phase, index=series.index)


def logistic_transition(spread, phase, states):
    s = pd.Series(states, index=spread.index)
    target = ((s.shift(-1) == 1) & (s == 0)).astype(int)
    df = pd.DataFrame({"target": target, "spread": spread, "phase": np.cos(phase)})
    df = df.dropna()
    if df.empty:
        return None, None
    model = Logit(df["target"], df[["spread", "phase"]]).fit(disp=False)
    fpr, tpr, _ = roc_curve(df["target"], model.predict())
    roc_auc = auc(fpr, tpr)
    return model, roc_auc


def compare_real_vs_nominal(real_price, nominal_price, states, ticker):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(real_price.index, real_price.values, label="Real")
    st = pd.Series(states, index=real_price.index)
    for val, col in [(0, "lightblue"), (1, "lightcoral")]:
        mask = st == val
        plt.fill_between(real_price.index, real_price.min(), real_price.max(), where=mask, color=col, alpha=0.3)
    plt.title(f"{ticker} Real Price")
    plt.subplot(1, 2, 2)
    plt.plot(nominal_price.index, nominal_price.values, label="Nominal")
    for val, col in [(0, "lightblue"), (1, "lightcoral")]:
        mask = st.reindex(nominal_price.index, method="nearest") == val
        plt.fill_between(nominal_price.index, nominal_price.min(), nominal_price.max(), where=mask, color=col, alpha=0.3)
    plt.title(f"{ticker} Nominal Price")
    plt.tight_layout()
    plt.savefig(f"{ticker}_real_vs_nominal.png")
    plt.close()


def plot_transition_surface(model, spread, phase, ticker):
    if model is None:
        return
    grid_s = np.linspace(spread.min(), spread.max(), 50)
    grid_p = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(grid_s, grid_p)
    logits = model.params["spread"] * X + model.params["phase"] * Y + model.params.get("const", 0)
    Z = 1 / (1 + np.exp(-logits))
    plt.figure(figsize=(6, 4))
    cs = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(cs)
    plt.xlabel("Spread")
    plt.ylabel("cos(phase)")
    plt.title("Transition Probability")
    plt.tight_layout()
    plt.savefig(f"{ticker}_transition_surface.png")
    plt.close()


def main(zip_path="mainz.zip"):
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    real = load_price_series(zip_path, real=True)
    nominal = load_price_series(zip_path, real=False)
    spreads = load_yield_spread(zip_path)
    if real.empty:
        raise ValueError("No price data")
    for ticker in real.columns[:1]:
        rprice = real[ticker].dropna()
        nprice = nominal.get(ticker, rprice)
        ret = np.log(rprice).diff().dropna()
        res, prob, states = fit_ms_garch(ret)
        compare_real_vs_nominal(rprice.loc[ret.index], nprice.loc[ret.index], states, ticker)
        vol = realized_vol(ret)
        phase = cycle_phase(vol)
        if not spreads.empty:
            country = ticker[:3]
            if country in spreads.columns:
                sp = spreads[country].reindex(ret.index, method="ffill")
                model, auc_val = logistic_transition(sp, phase, states)
                plot_transition_surface(model, sp, phase, ticker)
                if auc_val is not None:
                    with open(f"{ticker}_roc_auc.txt", "w") as f:
                        f.write(f"AUC: {auc_val:.3f}\n")

if __name__ == "__main__":
    main()

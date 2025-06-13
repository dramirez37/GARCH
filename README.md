# Volatility Cycle Analysis

This repository contains utilities for analyzing asset price volatility cycles.
The `volatility_cycle_analysis.py` script loads a CSV dataset, computes
realized volatility, detects dominant cycles and performs several
cross-asset analyses. Running the script now also produces basic
visualizations and summary tables that are useful when preparing results for
publication.

## Usage

```bash
python3 volatility_cycle_analysis.py
```

The script expects `mainz.zip` in the repo root. The archive should include a
CSV file for each ticker you wish to analyze. No extraction is necessary—the
script reads each file directly from the archive. These CSVs contain two
metadata rows followed by the header row at line 3, so the first line of price
data begins on row 4. The script automatically detects this layout.
When yield data are available, some datasets provide the 10‑year minus
3‑month spread directly (for example a ticker named `T10Y3M`) without the
underlying 3‑month series.  The helper in `ms_garch_full.py` loads these
spread tickers transparently and merges them with computed spreads from
10‑year and 3‑month yields for other countries.
The archive now also includes `IGT10Y3M`, which represents the United States
10‑year minus 3‑month Treasury spread.

Running the script writes summary tables and figures to the working directory,
including rolling correlation, cointegration and coherence heatmaps, PCA scree
plots and cluster assignments. Tickers are automatically grouped by the
"Series Type" metadata so that cross-asset analyses are performed both across
all tickers and within each asset group. When assets in a group have different
data ranges, the analysis starts at the first date shared by every member of
that group.


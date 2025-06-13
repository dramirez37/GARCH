# Volatility Cycle Analysis

This repository demonstrates a simplified pipeline for transforming raw
financial time series into a set of cleaned prices, volatility measures and
basic model results.  The structure roughly follows a nine step workflow:

1. `m1_make_timeframes.py` – convert mixed frequency data to monthly/weekly/daily
2. `m2_deseasonalize.py`  – compute returns and deseasonalize rolling volatility
3. `m3_cycle_detection.py` – detect dominant volatility cycles
4. `m4_arima.py`          – fit basic ARIMA models
5. `m5_garch_models.py`   – fit baseline GARCH models
6. `m6_macro_integration.py` – integrate macro spreads
7. `m7_cross_asset.py`    – cross asset correlations and PCA
8. `m8_bootstrap.py`      – bootstrap utilities
9. `m9_ml_models.py`      – placeholder for ML models

Raw CSV files live under `mainz/` while processed outputs are written to
`data/processed/`.  Figures are saved in `reports/figures/`.

Unit tests for a few helper utilities are in `tests/` and can be run with
`pytest`.

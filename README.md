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
script loads all of the files directly from the archive. Output CSV files and
figures are written to the working directory.


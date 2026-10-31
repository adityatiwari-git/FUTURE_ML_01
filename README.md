# Sales & Demand Forecasting | Future Interns ML Task 1

A reproducible machine-learning workflow that converts retail transactions into a daily sales series, explores demand patterns, and forecasts future sales.

## Project objective

Build a forecasting workflow that helps a retailer understand expected daily sales and plan business activity. The primary target is **daily gross sales revenue in GBP (£)**. Daily units sold are retained as a secondary business measure.

## Dataset

This project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online-retail), donated by Daqing Chen.

- 541,909 transaction rows.
- Transactions from 1 December 2010 to 9 December 2011.
- UK-based non-store online retailer.
- License: CC BY 4.0.
- DOI: https://doi.org/10.24432/C5BW33.

The original Excel workbook is downloaded locally by the preparation script and is not committed to this repository. See [data/README.md](data/README.md) for provenance and cleaning policy.

## Project structure

```text
src/
  prepare_data.py       # Download, validate, clean and aggregate transactions
  explore.py            # Exploratory analysis and business visuals
  train.py              # Time-series feature engineering and model evaluation
data/
  raw/                  # Downloaded source workbook (not committed)
  processed/            # Daily sales CSV (not committed)
reports/                # Metrics, predictions and charts (not committed)
requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the workflow

Run commands from the repository root:

```bash
python src/prepare_data.py
python src/explore.py
python src/train.py
```

The first command downloads the source workbook if it is not already present. An internet connection is required for this first run.

## Forecasting and evaluation approach

- Use time-ordered observations; do not randomly shuffle a time series.
- Compare a simple mean baseline, a same-weekday seasonal baseline, Random Forest and HistGradientBoosting.
- Create lag and rolling features using past observations only.
- Evaluate on a later chronological holdout period.
- Report MAE and RMSE in GBP, with R² as a supplementary metric.
- The test is a rolling one-day-ahead backtest: the actual observation is available before forecasting the next date. This is not a recursive multi-day forecast.

## Outputs

The scripts create:
- Clean daily data at `data/processed/daily_sales.csv`.
- Data preparation and EDA summaries under `reports/`.
- Monthly and weekday summaries as CSV files.
- Holdout predictions and model metrics.
- PNG charts in `reports/figures/`.

Generated data and reports are excluded from Git to keep the repository focused on source code and reproducibility.

## Interpretation and limitations

The dataset contains about one year of history, so annual seasonality cannot be robustly established from multiple years. The cleaning policy forecasts gross positive sales and excludes cancellations and non-positive sales lines; returns are not netted into that target. Transaction lines are treated as the observed demand record, but stockouts, unrecorded lost sales, inventory constraints, marketing campaigns and future business changes are not available in the source data.

This is an educational historical forecasting exercise, not a production sales-planning service. The backtest metrics describe only the selected historical holdout period and should not be presented as guaranteed future performance.

## Attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0.

# Sales & Demand Forecasting | Future Interns ML Task 1

A reproducible machine-learning project that converts retail transactions into a daily sales series, explores demand patterns, evaluates forecasting models, and produces a 30-day future sales forecast with business-facing insights.

## Project objective

Forecast daily gross sales revenue from historical retail transactions. The workflow covers data validation, cleaning, time-based aggregation, exploratory analysis, feature engineering, chronological model evaluation, recursive future forecasting, and business interpretation.

**Primary target:** daily gross sales revenue in GBP (£). Daily units sold are retained as a secondary business measure.

## Dataset

This project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online-retail), donated by Daqing Chen.

- 541,909 original transaction rows.
- Transactions from 1 December 2010 to 9 December 2011.
- UK-based non-store online retailer.
- License: CC BY 4.0.
- DOI: https://doi.org/10.24432/C5BW33.

The original Excel workbook is downloaded locally by the preparation script and is not committed to this repository. See [data/README.md](data/README.md) for provenance and cleaning policy.

## Project structure

```text
src/
  inspect_data.py       # Initial data-quality audit
  prepare_data.py       # Download, validate, clean and aggregate transactions
  explore.py            # Exploratory analysis and business visuals
  train.py              # Time-series features, evaluation and future forecast
data/
  raw/                  # Downloaded source workbook (not committed)
  processed/            # Daily sales CSV (not committed)
reports/                # Metrics, predictions, insights and charts (not committed)
requirements.txt
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the complete workflow

Run commands from the repository root:

```bash
python src/inspect_data.py
python src/prepare_data.py
python src/explore.py
python src/train.py
```

The preparation command downloads the source workbook if it is not already present. An internet connection is required for the first download.

## Methodology

### Data preparation

- Validate the expected transaction columns.
- Remove exact duplicate transaction rows.
- Exclude records missing essential invoice, date, quantity or price values.
- Exclude cancelled invoices (invoice numbers beginning with C).
- Exclude non-positive quantities and unit prices.
- Calculate transaction revenue as quantity multiplied by unit price.
- Aggregate revenue and units to calendar days, inserting zero-sales dates where needed.

The target is **gross positive sales**, not net revenue after returns. Customer IDs and product descriptions are not required for this aggregate forecasting task and are not used to remove otherwise valid transaction lines.

### Exploratory analysis

The EDA step creates daily trend, monthly revenue and weekday-average charts, along with CSV summaries and a JSON summary.

### Forecast features and models

Calendar features:
- Day of week, day of month, month, day of year and weekend indicator.

Historical features:
- Previous-day revenue (lag 1).
- Previous-week revenue (lag 7).
- Trailing 7-, 14- and 28-day average revenue, calculated using prior observations only.

Candidate models:
- Mean baseline.
- Same-weekday seasonal-naive baseline.
- Random Forest Regressor.
- Histogram-based Gradient Boosting Regressor.

### Evaluation

- Use a chronological 80/20 holdout; do not randomly shuffle time-series observations.
- Compare MAE and RMSE in GBP, with R² as a supplementary metric.
- The historical test is a **rolling one-day-ahead backtest**. Once a test-day prediction is made, that day's actual value is available for the next test-day prediction.
- Select the model with the lowest holdout MAE.

### Future forecast

The selected model is refitted using all available historical observations and used to generate a 30-day forecast. The forecast is recursive: each predicted day is appended to the history used to construct the next day's lag and rolling features. This is a genuine future forecast, but errors may accumulate across the horizon.

## Outputs

Generated locally (not committed):
- `data/processed/daily_sales.csv` — clean daily time series.
- `reports/data_summary.json` — data preparation summary.
- `reports/eda_summary.json` — exploratory summary.
- `reports/monthly_sales_summary.csv` and `reports/weekday_sales_summary.csv`.
- `reports/holdout_predictions.csv` — actual and holdout predictions for each model.
- `reports/metrics.json` — evaluation metrics and forecast summary.
- `reports/future_forecast.csv` — 30 daily future revenue estimates.
- `reports/business_insights.md` and `reports/business_insights.json` — business-facing interpretation and limitations.
- `reports/figures/` — EDA, holdout comparison and future forecast charts.

## Business interpretation and limitations

Forecasts can support short-term inventory and staffing discussions, but should be reviewed alongside promotions, holidays, stock availability and operational context. The dataset does not include stockouts, lost demand, marketing campaigns or future business changes.

The source contains roughly one year of history, so annual seasonality cannot be robustly established from multiple years. The backtest metrics describe one historical holdout period and are not guarantees of future performance. The recursive forecast is an estimate, not a production sales-planning service.

## Attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0.

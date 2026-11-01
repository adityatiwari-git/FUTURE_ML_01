# Sales & Demand Forecasting for Retail Businesses

**Future Interns — Machine Learning Internship | Task 1**

This project builds an end-to-end machine-learning workflow for analysing historical retail transactions and forecasting daily sales revenue. It takes raw transaction-level data, converts it into a clean daily time series, explores sales behaviour, trains and compares forecasting models, and generates a 30-day forecast with visualisations and business-oriented insights.

The project is designed as a practical demonstration of data cleaning, exploratory data analysis, time-series feature engineering, model evaluation and communicating results in a business context.

## What this project does

1. **Inspects the source data** — reports missing values, duplicate records, cancelled invoices, invalid quantities/prices and the date range.
2. **Prepares daily sales data** — cleans transaction rows, calculates line revenue and aggregates revenue and units sold by date.
3. **Explores sales patterns** — produces daily, monthly and weekday summaries and charts.
4. **Builds forecasting features** — creates calendar variables, lagged revenue and rolling averages using past observations.
5. **Trains and compares models** — evaluates simple baselines, Random Forest and Histogram-based Gradient Boosting.
6. **Evaluates performance chronologically** — uses a time-ordered holdout rather than randomly shuffling dates.
7. **Generates a 30-day forecast** — refits the selected model on the available history and recursively predicts the next 30 days.
8. **Communicates findings** — saves metrics, prediction files, charts and business-facing insights.

## Dataset

The project uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online-retail), donated by Daqing Chen.

- **Original size:** 541,909 transaction rows
- **Period:** 1 December 2010 to 9 December 2011
- **Business context:** a UK-based, non-store online retailer
- **License:** CC BY 4.0
- **DOI:** https://doi.org/10.24432/C5BW33

The preparation script downloads the Excel workbook into `data/raw/` when it is not already present. The raw workbook and generated datasets are intentionally not committed to GitHub. Dataset provenance and cleaning decisions are documented in [data/README.md](data/README.md).

## Forecast target

The primary prediction target is **daily gross positive sales revenue in GBP (£)**. Daily units sold are also retained as a supporting business measure.

Cancelled invoices and rows with missing essential fields, non-positive quantities or non-positive unit prices are excluded. Revenue is calculated as quantity multiplied by unit price and summed by calendar day. Dates without recorded sales are represented as zero-sales days.

This is a gross-sales target: returns are not netted against revenue. Therefore, the forecast should not be interpreted as net revenue or profit.

## Repository structure

```text
FUTURE_ML_01/
├── .github/workflows/       # Automated repository validation
├── data/
│   ├── raw/                 # Downloaded source workbook (not committed)
│   ├── processed/           # Prepared daily data (not committed)
│   └── README.md            # Dataset source and cleaning notes
├── reports/                 # Generated metrics, CSVs, insights and charts
├── src/
│   ├── inspect_data.py      # Initial data-quality inspection
│   ├── prepare_data.py      # Cleaning and daily aggregation
│   ├── explore.py           # Exploratory analysis and visualisations
│   └── train.py             # Feature engineering, model evaluation and forecast
├── tests/
│   └── test_forecasting.py  # Forecasting logic and leakage checks
├── requirements.txt
└── README.md
```

Generated data and reports are created locally and are excluded from version control to avoid committing the full source workbook and reproducible output files.

## Tools and libraries

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- OpenPyXL
- Git and GitHub Actions

## Setup

Python 3.10 or later is recommended.

```bash
git clone https://github.com/adityatiwari-git/FUTURE_ML_01.git
cd FUTURE_ML_01
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the project

Run each command from the repository root, in this order:

```bash
python src/inspect_data.py
python src/prepare_data.py
python src/explore.py
python src/train.py
```

The first preparation run downloads the UCI Excel workbook and requires an internet connection. Later runs reuse the local workbook.

Run the automated unit tests:

```bash
python -m unittest discover -s tests -v
```

The tests cover important forecasting behaviour, including use of past-only lag/rolling features, recursive forecast generation and protection against future-target leakage in the seasonal baseline.

## Forecasting approach

### Feature engineering

The daily time series is augmented with:

- Calendar features: weekday, day of month, month, day of year and weekend indicator.
- Lag features: previous-day revenue (lag 1) and previous-week revenue (lag 7).
- Rolling features: 7-, 14- and 28-day averages calculated from earlier observations only.

Using shifted rolling windows prevents the current day's target from leaking into its own features.

### Models compared

- Mean baseline
- Same-weekday seasonal-naive baseline
- Random Forest Regressor
- Histogram-based Gradient Boosting Regressor

### Evaluation strategy

The data is kept in chronological order and divided into an 80/20 holdout. The holdout contains 70 test days after the feature warm-up period. Performance is measured using:

- **MAE:** mean absolute error in GBP; lower is better.
- **RMSE:** root mean squared error in GBP; penalises larger errors more strongly.
- **R²:** supplementary measure of explained variation; higher is generally better.

The historical evaluation is a **rolling one-day-ahead backtest**. After a prediction is made for a test date, that date's actual observation becomes available to construct features for the following test date. This is different from predicting the entire 70-day holdout recursively.

The model is selected by the lowest holdout MAE. In the local run documented during development, Random Forest achieved an MAE of approximately **£13,406.42**, while Gradient Boosting achieved a slightly lower RMSE of approximately **£23,560.71** and R² of **0.4422**. These are results from one historical holdout period, not a guarantee of future accuracy.

### 30-day forecast

After evaluation, the selected Random Forest model is refitted using all available historical observations. It then predicts the next 30 days recursively: each predicted value is added to the history used to calculate features for the next date.

The local run generated a forecast from **10 December 2011 to 8 January 2012**. The output showed a pronounced repeating weekday pattern and some unusually low/high daily estimates. These values should be treated cautiously and reviewed before any operational use. Recursive forecasts can accumulate error, and this dataset contains only about one year of history.

## Generated outputs

After running the workflow, the following files are produced locally:

| Output | Purpose |
|---|---|
| `data/processed/daily_sales.csv` | Clean daily revenue and sales measures |
| `reports/data_summary.json` | Data preparation summary |
| `reports/eda_summary.json` | Key exploratory findings |
| `reports/monthly_sales_summary.csv` | Monthly sales summary |
| `reports/weekday_sales_summary.csv` | Weekday sales summary |
| `reports/holdout_predictions.csv` | Actual values and holdout predictions |
| `reports/metrics.json` | Model metrics and forecast metadata |
| `reports/future_forecast.csv` | 30-day daily revenue estimates |
| `reports/business_insights.md` | Business interpretation and limitations |
| `reports/figures/` | EDA, model comparison and forecast charts |

Important charts include:

- `daily_sales_trend.png`
- `monthly_sales.png`
- `weekday_sales.png`
- `actual_vs_predicted.png`
- `model_comparison_mae.png`
- `future_sales_forecast.png`

## Business relevance and limitations

Daily sales forecasts can help a business begin planning inventory, staffing and short-term operations. The charts and weekday/monthly summaries help communicate patterns that may be useful to business teams.

However, this dataset does not include information about stockouts, promotions, marketing campaigns, product availability, costs or future business changes. It covers roughly one year, so it cannot establish robust annual seasonality from multiple years of observations. The model forecasts historical gross sales patterns; it does not estimate lost demand, profit or net sales after returns.

The 30-day forecast is an educational project output and should be validated against additional data and business context before being used for real purchasing or staffing decisions.

## Attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0.

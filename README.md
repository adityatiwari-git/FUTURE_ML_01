# Sales & Demand Forecasting | Future Interns ML Task 1

Forecast product demand from historical daily sales using a reproducible machine-learning workflow.

## Project status
Starter implementation committed. The included demo dataset is **synthetic** and is intended to verify the pipeline; it is not evidence of real business performance. Replace it with a properly licensed business dataset before drawing commercial conclusions.

## Quick start
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

The script generates synthetic daily sales, builds calendar/lag/rolling features, trains a baseline and a random-forest regressor, evaluates on a chronological holdout set, and saves predictions and metrics under `reports/`.

## Method
- Target: daily units sold for one demo product/store series.
- Features: calendar fields, price, promotion flag, lagged demand, and rolling means.
- Validation: chronological split (no random shuffling).
- Metrics: MAE, RMSE, and R². Interpret metrics in the context of the synthetic data.

## Structure
```
src/train.py          # end-to-end reproducible demo
data/                 # generated locally; not committed
reports/              # generated metrics and predictions
requirements.txt
```

## Next steps
1. Select a documented, licensed real-world sales dataset.
2. Adapt the data-loading and feature-building steps to its schema.
3. Compare against seasonal-naive and other forecasting baselines.
4. Tune only on training/validation periods; preserve a final untouched test period.
5. Document dataset provenance, limitations, and business implications.

## Limitations
This starter is a supervised regression demonstration, not a production forecasting service. Forecast quality depends on data volume, seasonality, promotions, stockouts, and changes in business conditions.

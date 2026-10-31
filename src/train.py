"""Train and evaluate daily retail revenue forecasting models.

Run after prepare_data.py:
    python src/train.py

The holdout is the final chronological 20% of observations. All lagged
features are shifted so a prediction for day t only uses data through t-1.
The test is a rolling one-day-ahead backtest: actual observations become
available after each prediction.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_sales.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TARGET = "revenue_gbp"
FEATURES = [
    "day_of_week", "day_of_month", "month", "day_of_year", "is_weekend",
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
]


def load_series():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Daily dataset not found. Run 'python src/prepare_data.py' first."
        )
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    missing = {"date", TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Daily dataset is missing columns: {sorted(missing)}")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")
    if len(df) < 70:
        raise ValueError("At least 70 daily records are required.")
    return df


def make_features(df):
    out = df.copy()
    dates = out["date"]
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_month"] = dates.dt.day
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    out["lag_1"] = out[TARGET].shift(1)
    out["lag_7"] = out[TARGET].shift(7)
    prior_values = out[TARGET].shift(1)
    for window in (7, 14, 28):
        out[f"rolling_mean_{window}"] = prior_values.rolling(
            window=window, min_periods=window
        ).mean()

    return out.dropna(subset=FEATURES).reset_index(drop=True)


def metric_values(actual, predicted):
    return {
        "mae_gbp": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse_gbp": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def seasonal_naive_rolling(raw, test_df):
    """Predict with the last observed value from the same weekday."""
    history = raw.set_index("date")["revenue_gbp"].copy()
    predictions = []
    for _, row in test_df.iterrows():
        date = row["date"]
        same_weekday = history[history.index.dayofweek == date.dayofweek]
        prediction = (
            float(same_weekday.iloc[-1])
            if not same_weekday.empty else float(history.iloc[-1])
        )
        predictions.append(max(0.0, prediction))
        history.loc[date] = float(row["revenue_gbp"])
    return np.asarray(predictions)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_series()
    data = make_features(raw)
    split = int(len(data) * 0.8)
    train_df = data.iloc[:split].copy()
    test_df = data.iloc[split:].copy()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET].to_numpy()

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=3, max_features=0.9,
            random_state=42, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42,
        ),
    }

    predictions = {}
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = np.maximum(0.0, model.predict(X_test))
        predictions[name] = predicted
        metrics[name] = metric_values(y_test, predicted)

    predictions["seasonal_naive_7day"] = seasonal_naive_rolling(
        raw, test_df
    )
    metrics["seasonal_naive_7day"] = metric_values(
        y_test, predictions["seasonal_naive_7day"]
    )

    model_names = list(models) + ["seasonal_naive_7day"]
    best_name = min(model_names, key=lambda name: metrics[name]["mae_gbp"])

    metrics["evaluation"] = {
        "target": "daily gross revenue (GBP)",
        "total_daily_observations": int(len(raw)),
        "modelled_observations_after_lag_warmup": int(len(data)),
        "training_observations": int(len(train_df)),
        "testing_observations": int(len(test_df)),
        "train_end_date": str(train_df["date"].max().date()),
        "test_start_date": str(test_df["date"].min().date()),
        "test_end_date": str(test_df["date"].max().date()),
        "split": "Chronological 80/20 holdout",
        "test_protocol": (
            "Rolling one-day-ahead backtest. Each forecast uses actual "
            "historical observations available before that date; after "
            "forecasting a date, its actual value is added to history."
        ),
        "selected_model_by_mae": best_name,
        "limitation": (
            "This is a historical one-day-ahead backtest, not a recursive "
            "multi-day production forecast."
        ),
    }

    output = test_df[["date", TARGET]].copy()
    for name, values in predictions.items():
        output[f"{name}_prediction"] = np.round(values, 2)
    output.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )

    plt.figure(figsize=(13, 5))
    plt.plot(test_df["date"], y_test, label="Actual", linewidth=1.2)
    plt.plot(
        test_df["date"], predictions[best_name],
        label=f"Predicted ({best_name})", linewidth=1.1
    )
    plt.title("Actual vs Predicted Daily Gross Revenue — Holdout")
    plt.xlabel("Date")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, [metrics[name]["mae_gbp"] for name in model_names])
    plt.title("Forecast Model Comparison — Mean Absolute Error")
    plt.xlabel("Model")
    plt.ylabel("MAE (£), lower is better")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_mae.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=4))
    print(f"Selected model by MAE: {best_name}")


if __name__ == "__main__":
    main()

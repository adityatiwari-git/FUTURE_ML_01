"""Train, evaluate, and forecast daily retail sales.

Run from the repository root after preparing the data:
    python src/prepare_data.py
    python src/explore.py
    python src/train.py

The holdout evaluation is a rolling one-day-ahead backtest. Future forecasts
are generated recursively, so later forecast days use earlier predictions
rather than unavailable actual values.
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
FORECAST_HORIZON_DAYS = 30
FEATURES = [
    "day_of_week", "day_of_month", "month", "day_of_year", "is_weekend",
    "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
]


def load_series():
    """Load and validate the daily revenue time series."""
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
    if not df["date"].is_monotonic_increasing:
        raise ValueError("Dates must be in ascending order.")
    if len(df) < 70:
        raise ValueError("At least 70 daily records are required.")
    return df


def make_features(df):
    """Create calendar, lag and rolling features using past values only."""
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
    """Return standard regression metrics in business-readable units."""
    return {
        "mae_gbp": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse_gbp": round(float(mean_squared_error(actual, predicted) ** 0.5), 2),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def seasonal_naive_rolling(raw, test_df):
    """Predict each test day using the latest observed value from its weekday."""
    first_test_date = test_df["date"].min()
    history = (
        raw.loc[raw["date"] < first_test_date]
        .set_index("date")[TARGET]
        .copy()
    )
    predictions = []
    for _, row in test_df.iterrows():
        date = row["date"]
        same_weekday = history[history.index.dayofweek == date.dayofweek]
        prediction = (
            float(same_weekday.iloc[-1])
            if not same_weekday.empty else float(history.iloc[-1])
        )
        predictions.append(max(0.0, prediction))
        history.loc[date] = float(row[TARGET])
    return np.asarray(predictions)


def build_models():
    """Return the candidate models used for the holdout comparison."""
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def future_feature_row(history, date):
    """Build one future feature row from observations available before date."""
    values = history.sort_index()
    if len(values) < 28:
        raise ValueError("At least 28 historical days are required to forecast.")

    row = {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),
        "lag_1": float(values.iloc[-1]),
        "lag_7": float(values.iloc[-7]),
        "rolling_mean_7": float(values.iloc[-7:].mean()),
        "rolling_mean_14": float(values.iloc[-14:].mean()),
        "rolling_mean_28": float(values.iloc[-28:].mean()),
    }
    return pd.DataFrame([row], columns=FEATURES)


def recursive_forecast(model_name, fitted_model, raw, horizon):
    """Forecast future dates recursively without using future actual values."""
    history = raw.set_index("date")[TARGET].astype(float).copy()
    future_dates = pd.date_range(
        history.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D"
    )
    forecasts = []

    for date in future_dates:
        if model_name == "seasonal_naive_7day":
            same_weekday = history[history.index.dayofweek == date.dayofweek]
            prediction = (
                float(same_weekday.iloc[-1])
                if not same_weekday.empty else float(history.iloc[-1])
            )
        else:
            features = future_feature_row(history, date)
            prediction = float(fitted_model.predict(features)[0])

        prediction = max(0.0, prediction)
        forecasts.append(prediction)
        # Append the prediction so the next forecast day uses predicted history.
        history.loc[date] = prediction

    return pd.DataFrame({
        "date": future_dates,
        "forecast_revenue_gbp": np.round(forecasts, 2),
        "model": model_name,
    })


def write_business_summary(metrics, forecast, eda_summary):
    """Create a concise business-facing interpretation from actual outputs."""
    evaluation = metrics["evaluation"]
    best_name = evaluation["selected_model_by_mae"]
    best_metrics = metrics[best_name]
    forecast_total = round(float(forecast["forecast_revenue_gbp"].sum()), 2)
    forecast_average = round(float(forecast["forecast_revenue_gbp"].mean()), 2)

    summary = {
        "forecast_target": "daily gross sales revenue",
        "currency": "GBP",
        "selected_model": best_name,
        "holdout_mae_gbp": best_metrics["mae_gbp"],
        "holdout_rmse_gbp": best_metrics["rmse_gbp"],
        "holdout_r2": best_metrics["r2"],
        "forecast_horizon_days": int(len(forecast)),
        "forecast_start_date": str(forecast["date"].min().date()),
        "forecast_end_date": str(forecast["date"].max().date()),
        "forecast_total_revenue_gbp": forecast_total,
        "forecast_average_daily_revenue_gbp": forecast_average,
        "historical_average_daily_revenue_gbp": eda_summary.get(
            "average_daily_revenue_gbp"
        ),
        "historical_highest_revenue_month": eda_summary.get(
            "highest_revenue_month"
        ),
        "interpretation": (
            "The forecast is a model-based estimate for planning, not a "
            "guarantee. The holdout metrics describe a historical rolling "
            "one-day-ahead test and may not match future performance."
        ),
        "business_use": [
            "Use the daily estimates as one input to short-term staffing and "
            "inventory discussions.",
            "Review unusually high or low forecast days against promotions, "
            "holidays and operational events before acting.",
            "Compare forecasts with actual sales as new observations arrive "
            "and retrain when the business pattern changes.",
        ],
        "limitations": [
            "The source contains roughly one year of transaction history.",
            "The target is gross positive sales; returns are not netted against it.",
            "The dataset does not include stockouts, lost demand, promotions or "
            "future operational changes.",
            "The future forecast is recursive, so errors can accumulate across "
            "the forecast horizon.",
        ],
    }

    (REPORT_DIR / "business_insights.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )

    lines = [
        "# Sales Forecast — Business Insights",
        "",
        f"- **Forecast target:** Daily gross sales revenue (GBP)",
        f"- **Selected model:** {best_name}",
        f"- **Holdout MAE:** £{best_metrics['mae_gbp']:,.2f}",
        f"- **Holdout RMSE:** £{best_metrics['rmse_gbp']:,.2f}",
        f"- **Holdout R²:** {best_metrics['r2']:.4f}",
        f"- **Forecast period:** {summary['forecast_start_date']} to "
        f"{summary['forecast_end_date']}",
        f"- **Forecast total:** £{forecast_total:,.2f}",
        f"- **Forecast daily average:** £{forecast_average:,.2f}",
        "",
        "## Business interpretation",
        "",
        summary["interpretation"],
        "",
        "## Suggested use",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["business_use"])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    (REPORT_DIR / "business_insights.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


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

    models = build_models()
    predictions = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = np.maximum(0.0, model.predict(X_test))
        predictions[name] = predicted
        metrics[name] = metric_values(y_test, predicted)

    predictions["seasonal_naive_7day"] = seasonal_naive_rolling(raw, test_df)
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
            "Rolling one-day-ahead backtest. Each prediction uses actual "
            "historical observations available before that date; after "
            "forecasting a date, its actual value is added to history."
        ),
        "selected_model_by_mae": best_name,
        "future_forecast_protocol": (
            "Recursive multi-day forecast. Each predicted value is appended "
            "to history and used to construct features for the next date."
        ),
        "limitation": (
            "The holdout backtest is one-day-ahead, while the future forecast "
            "is recursive. These are different evaluation settings."
        ),
    }

    holdout = test_df[["date", TARGET]].copy()
    for name, values in predictions.items():
        holdout[f"{name}_prediction"] = np.round(values, 2)
    holdout.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)

    # Refit the selected machine-learning model on all available history.
    final_model = None
    if best_name in models:
        final_model = build_models()[best_name]
        final_model.fit(data[FEATURES], data[TARGET])

    forecast = recursive_forecast(
        best_name, final_model, raw, FORECAST_HORIZON_DAYS
    )
    forecast.to_csv(REPORT_DIR / "future_forecast.csv", index=False)

    metrics["future_forecast"] = {
        "horizon_days": FORECAST_HORIZON_DAYS,
        "start_date": str(forecast["date"].min().date()),
        "end_date": str(forecast["date"].max().date()),
        "total_forecast_revenue_gbp": round(
            float(forecast["forecast_revenue_gbp"].sum()), 2
        ),
        "average_daily_forecast_revenue_gbp": round(
            float(forecast["forecast_revenue_gbp"].mean()), 2
        ),
    }
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=4), encoding="utf-8"
    )

    # Actual versus predicted values on the historical holdout.
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

    # Compare models using MAE; lower is better.
    plt.figure(figsize=(10, 5))
    plt.bar(model_names, [metrics[name]["mae_gbp"] for name in model_names])
    plt.title("Forecast Model Comparison — Mean Absolute Error")
    plt.xlabel("Model")
    plt.ylabel("MAE (£), lower is better")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "model_comparison_mae.png", dpi=160)
    plt.close()

    # Show the recent historical context and the recursive future forecast.
    recent = raw.tail(90)
    plt.figure(figsize=(13, 5))
    plt.plot(
        recent["date"], recent[TARGET],
        label="Historical daily revenue", linewidth=1.2
    )
    plt.plot(
        forecast["date"], forecast["forecast_revenue_gbp"],
        label=f"{FORECAST_HORIZON_DAYS}-day forecast ({best_name})",
        linewidth=1.4
    )
    plt.axvline(raw["date"].max(), linestyle="--", linewidth=1)
    plt.title("Daily Sales Revenue — History and Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue (£)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "future_sales_forecast.png", dpi=160)
    plt.close()

    eda_path = REPORT_DIR / "eda_summary.json"
    eda_summary = (
        json.loads(eda_path.read_text(encoding="utf-8"))
        if eda_path.exists() else {}
    )
    write_business_summary(metrics, forecast, eda_summary)

    print("Model evaluation and future forecasting completed.")
    print(json.dumps(metrics, indent=4))
    print(f"Selected model by MAE: {best_name}")
    print(f"Future forecast: {REPORT_DIR / 'future_forecast.csv'}")
    print(f"Business insights: {REPORT_DIR / 'business_insights.md'}")


if __name__ == "__main__":
    main()

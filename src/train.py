"""Reproducible synthetic sales forecasting baseline.

Run from the repository root: python src/train.py
The generated data is synthetic and must not be represented as real sales data.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, REPORT_DIR = ROOT / "data", ROOT / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def make_demo_data(n_days: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    day = np.arange(n_days)
    promo = rng.binomial(1, 0.12, n_days)
    price = 20 + 2 * np.sin(2 * np.pi * day / 365) + rng.normal(0, 0.6, n_days)
    weekly = 8 * np.sin(2 * np.pi * day / 7)
    annual = 12 * np.sin(2 * np.pi * day / 365)
    trend = 0.015 * day
    noise = rng.normal(0, 4, n_days)
    demand = np.maximum(0, 45 + weekly + annual + trend + 18 * promo - 1.2 * (price - 20) + noise)
    return pd.DataFrame({"date": dates, "units_sold": np.round(demand).astype(int),
                         "price": price.round(2), "promotion": promo})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    out["lag_1"] = out["units_sold"].shift(1)
    out["lag_7"] = out["units_sold"].shift(7)
    out["rolling_mean_7"] = out["units_sold"].shift(1).rolling(7).mean()
    return out.dropna().reset_index(drop=True)


def main() -> None:
    raw = make_demo_data()
    raw.to_csv(DATA_DIR / "synthetic_daily_sales.csv", index=False)
    data = add_features(raw)
    features = ["price", "promotion", "day_of_week", "month", "day_of_year",
                "lag_1", "lag_7", "rolling_mean_7"]
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[features], train["units_sold"]
    X_test, y_test = test[features], test["units_sold"]

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                random_state=42, n_jobs=-1),
    }
    results = {}
    prediction_frame = test[["date", "units_sold"]].copy()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = np.maximum(0, model.predict(X_test))
        prediction_frame[name + "_prediction"] = pred.round(2)
        results[name] = {
            "mae": round(mean_absolute_error(y_test, pred), 3),
            "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 3),
            "r2": round(r2_score(y_test, pred), 3),
        }
    prediction_frame.to_csv(REPORT_DIR / "holdout_predictions.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved demo data to {DATA_DIR} and evaluation to {REPORT_DIR}")


if __name__ == "__main__":
    main()

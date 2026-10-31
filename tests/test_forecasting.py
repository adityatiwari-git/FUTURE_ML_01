import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train


class ForecastingFeatureTests(unittest.TestCase):
    def test_lag_and_rolling_features_use_past_values(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        frame = pd.DataFrame({
            "date": dates,
            "revenue_gbp": np.arange(1, 41, dtype=float),
        })

        featured = train.make_features(frame)
        first = featured.iloc[0]

        self.assertEqual(first["date"], dates[28])
        self.assertEqual(first["lag_1"], 28.0)
        self.assertEqual(first["lag_7"], 22.0)
        self.assertAlmostEqual(first["rolling_mean_7"], 25.0)
        self.assertAlmostEqual(first["rolling_mean_14"], 21.5)
        self.assertAlmostEqual(first["rolling_mean_28"], 14.5)

    def test_recursive_forecast_returns_requested_horizon(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        history = pd.DataFrame({
            "date": dates,
            "revenue_gbp": np.arange(100, 140, dtype=float),
        })
        model = DummyRegressor(strategy="mean")
        model.fit(np.zeros((10, len(train.FEATURES))), np.arange(10, dtype=float))

        forecast = train.recursive_forecast(
            "mean_baseline", model, history, horizon=5
        )

        self.assertEqual(len(forecast), 5)
        self.assertEqual(
            forecast["date"].iloc[0],
            dates[-1] + pd.Timedelta(days=1),
        )
        self.assertTrue((forecast["forecast_revenue_gbp"] >= 0).all())
        self.assertEqual(forecast["model"].unique().tolist(), ["mean_baseline"])


    def test_seasonal_backtest_does_not_read_future_test_targets(self):
        dates = pd.date_range("2024-01-01", periods=42, freq="D")
        raw = pd.DataFrame({
            "date": dates,
            "revenue_gbp": np.arange(1, 43, dtype=float),
        })
        test_df = raw.tail(7).copy()

        predictions = train.seasonal_naive_rolling(raw, test_df)

        # Each first-week forecast must use the matching weekday from
        # the preceding week, not the actual value from the test period.
        np.testing.assert_allclose(predictions, np.arange(29, 36, dtype=float))

    def test_seasonal_naive_forecast_uses_same_weekday(self):
        dates = pd.date_range("2024-01-01", periods=35, freq="D")
        history = pd.DataFrame({
            "date": dates,
            "revenue_gbp": np.arange(1, 36, dtype=float),
        })

        forecast = train.recursive_forecast(
            "seasonal_naive_7day", None, history, horizon=7
        )

        expected = history["revenue_gbp"].tail(7).to_numpy()
        np.testing.assert_allclose(
            forecast["forecast_revenue_gbp"].to_numpy(), expected
        )


if __name__ == "__main__":
    unittest.main()

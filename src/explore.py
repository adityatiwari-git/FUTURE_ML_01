"""Exploratory analysis for the prepared daily retail sales series."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_sales.csv"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def load_daily_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Daily dataset not found. Run 'python src/prepare_data.py' first."
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    required = {"date", "revenue_gbp", "units_sold"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Daily data is missing columns: {sorted(missing)}")

    return df.sort_values("date").reset_index(drop=True)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_daily_data()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["day_of_week"] = df["date"].dt.day_name()

    monthly = (
        df.groupby("month", as_index=False)
        .agg(revenue_gbp=("revenue_gbp", "sum"),
             units_sold=("units_sold", "sum"))
    )
    weekday = (
        df.groupby("day_of_week", as_index=False)
        .agg(revenue_gbp=("revenue_gbp", "mean"),
             units_sold=("units_sold", "mean"))
    )
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    weekday["day_of_week"] = pd.Categorical(
        weekday["day_of_week"], categories=weekday_order, ordered=True
    )
    weekday = weekday.sort_values("day_of_week")

    summary = {
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "number_of_days": int(len(df)),
        "days_with_sales": int((df["revenue_gbp"] > 0).sum()),
        "total_revenue_gbp": round(float(df["revenue_gbp"].sum()), 2),
        "average_daily_revenue_gbp": round(float(df["revenue_gbp"].mean()), 2),
        "median_daily_revenue_gbp": round(float(df["revenue_gbp"].median()), 2),
        "maximum_daily_revenue_gbp": round(float(df["revenue_gbp"].max()), 2),
        "highest_revenue_day": str(
            df.loc[df["revenue_gbp"].idxmax(), "date"].date()
        ),
        "highest_revenue_month": str(
            monthly.loc[monthly["revenue_gbp"].idxmax(), "month"].to_period("M")
        ),
        "best_average_revenue_weekday": str(
            weekday.loc[weekday["revenue_gbp"].idxmax(), "day_of_week"]
        ),
        "revenue_unit": "GBP",
    }

    df.to_csv(REPORT_DIR / "daily_sales_with_calendar.csv", index=False)
    monthly.to_csv(REPORT_DIR / "monthly_sales_summary.csv", index=False)
    weekday.to_csv(REPORT_DIR / "weekday_sales_summary.csv", index=False)
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )

    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["revenue_gbp"], linewidth=0.8)
    plt.title("Daily Gross Sales Revenue")
    plt.xlabel("Date")
    plt.ylabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "daily_sales_trend.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(monthly["month"].dt.strftime("%Y-%m"), monthly["revenue_gbp"])
    plt.title("Monthly Gross Sales Revenue")
    plt.xlabel("Month")
    plt.ylabel("Revenue (£)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "monthly_sales.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(weekday["day_of_week"].astype(str), weekday["revenue_gbp"])
    plt.title("Average Daily Revenue by Weekday")
    plt.xlabel("Day of week")
    plt.ylabel("Average revenue (£)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "weekday_sales.png", dpi=160)
    plt.close()

    print("Exploratory analysis completed.")
    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()

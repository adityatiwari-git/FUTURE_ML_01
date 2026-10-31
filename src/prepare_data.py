"""Download and prepare the UCI Online Retail transaction data.

Run from the repository root:
    python src/prepare_data.py

The raw workbook is downloaded locally and is not committed to Git.
"""
from pathlib import Path
import json
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00352/Online%20Retail.xlsx"
)
RAW_FILE = RAW_DIR / "online_retail.xlsx"

REQUIRED_COLUMNS = {
    "InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"
}


def download_dataset() -> Path:
    """Download the original workbook if it is not already available."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 0:
        print(f"Using existing dataset: {RAW_FILE}")
        return RAW_FILE

    print("Downloading the UCI Online Retail workbook...")
    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0 sales-forecasting-student-project"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI workbook. Check your internet "
            "connection and download Online Retail.xlsx manually from "
            "https://archive.ics.uci.edu/dataset/352/online-retail"
        ) from exc

    if len(data) < 1_000_000:
        raise RuntimeError(
            "The downloaded file is unexpectedly small. The UCI page may "
            f"have returned an error page (Content-Type: {content_type})."
        )

    RAW_FILE.write_bytes(data)
    print(f"Downloaded {len(data):,} bytes to {RAW_FILE}")
    return RAW_FILE


def load_transactions(path: Path) -> pd.DataFrame:
    """Read the workbook and validate the expected transaction schema."""
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as exc:
        raise RuntimeError(
            f"Could not read {path}. Confirm it is the original Excel file."
        ) from exc

    df.columns = [str(column).strip() for column in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )
    return df


def prepare_daily_sales(raw: pd.DataFrame):
    """Clean transaction lines and aggregate gross sales by calendar day."""
    df = raw.copy()
    original_rows = len(df)
    
    # Remove exact duplicate transaction records
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    # Remove records that cannot contribute to a valid sales calculation.
    df = df.dropna(
        subset=["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    ).copy()
    rows_after_required_field_validation = int(len(df))

    # UCI documents invoices beginning with C as cancellations.
    cancelled_mask = (
        df["InvoiceNo"].str.upper().str.startswith("C", na=False)
    )
    cancelled_rows = int(cancelled_mask.sum())
    df = df.loc[~cancelled_mask].copy()

    # This project forecasts gross positive sales, not returns/refunds.
    invalid_sales_mask = (df["Quantity"] <= 0) | (df["UnitPrice"] <= 0)
    invalid_sales_rows = int(invalid_sales_mask.sum())
    df = df.loc[~invalid_sales_mask].copy()

    df["revenue_gbp"] = df["Quantity"] * df["UnitPrice"]
    df["date"] = df["InvoiceDate"].dt.normalize()

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            revenue_gbp=("revenue_gbp", "sum"),
            units_sold=("Quantity", "sum"),
            transaction_lines=("InvoiceNo", "size"),
            unique_invoices=("InvoiceNo", "nunique"),
        )
        .sort_values("date")
    )

    # Make the time index continuous, including days without transactions.
    full_dates = pd.date_range(
        daily["date"].min(), daily["date"].max(), freq="D"
    )
    daily = (
        daily.set_index("date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    integer_columns = [
        "units_sold", "transaction_lines", "unique_invoices"
    ]
    for column in integer_columns:
        daily[column] = daily[column].astype("int64")
    daily["revenue_gbp"] = daily["revenue_gbp"].astype(float).round(2)

    summary = {
        "dataset": "UCI Online Retail",
        "dataset_id": 352,
        "source_url": "https://archive.ics.uci.edu/dataset/352/online-retail",
        "citation": (
            "Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5BW33."
        ),
        "license": "CC BY 4.0",
        "original_rows": int(original_rows),
        "rows_after_required_field_validation": rows_after_required_field_validation,
        "duplicate_rows_removed": duplicate_rows,
        "rows_after_duplicate_removal": int(original_rows - duplicate_rows),
        "cancelled_invoice_rows_removed": cancelled_rows,
        "non_positive_sales_rows_removed": invalid_sales_rows,
        "clean_transaction_rows": int(len(df)),
        "daily_records": int(len(daily)),
        "date_start": str(daily["date"].min().date()),
        "date_end": str(daily["date"].max().date()),
        "total_gross_revenue_gbp": round(
            float(daily["revenue_gbp"].sum()), 2
        ),
        "total_units_sold": int(daily["units_sold"].sum()),
        "cleaning_policy": (
            "Exact duplicate rows, cancelled invoices and rows with missing "
            "essential values were excluded. Non-positive quantity or unit "
            "price rows were excluded. CustomerID and Description were not "
            "required. The target represents gross positive sales; returns "
            "are not netted against gross revenue."
        ),
    }
    return daily, summary


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = download_dataset()
    raw = load_transactions(dataset_path)
    daily, summary = prepare_daily_sales(raw)

    output_path = PROCESSED_DIR / "daily_sales.csv"
    summary_path = REPORT_DIR / "data_summary.json"

    daily.to_csv(output_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )

    print("\nData preparation completed successfully.")
    print(json.dumps(summary, indent=4))
    print(f"Daily data: {output_path}")
    print(f"Data summary: {summary_path}")


if __name__ == "__main__":
    main()

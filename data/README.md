# Dataset

This project uses the **Online Retail** dataset from the UCI Machine Learning Repository.

- Dataset page: https://archive.ics.uci.edu/dataset/352/online-retail
- DOI: https://doi.org/10.24432/C5BW33
- Original donor: Daqing Chen
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Original workbook: `Online Retail.xlsx` (downloaded locally; not tracked in Git)

## Local workflow

Run:

```bash
python src/prepare_data.py
```

The script downloads the original workbook into `data/raw/` and generates `data/processed/daily_sales.csv`. Both directories are excluded from version control so that large source data and generated files are not committed.

## Data handling

The source records transactions for a UK-based online retailer from 1 December 2010 through 9 December 2011. Invoice numbers beginning with `C` identify cancellations. The preparation script removes cancellations, missing required values and non-positive quantity/price lines. The resulting revenue target is gross positive sales in GBP; returns are not netted into that target.

## Citation

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.

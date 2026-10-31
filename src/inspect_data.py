import pandas as pd

# Load the original dataset
file_path = "data/raw/online_retail.xlsx"
df = pd.read_excel(file_path)

print("DATA QUALITY REPORT")
print("-" * 40)

# 1. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 2. Duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# 3. Cancelled invoices
cancelled = df["InvoiceNo"].astype(str).str.startswith("C")
print("\nCancelled invoice records:", cancelled.sum())

# 4. Invalid quantities
print("Zero or negative quantities:", (df["Quantity"] <= 0).sum())

# 5. Invalid prices
print("Zero or negative prices:", (df["UnitPrice"] <= 0).sum())

# 6. Date range
print("\nDate range:")
print("Start:", df["InvoiceDate"].min())
print("End:", df["InvoiceDate"].max())
import pandas as pd

# Load the dataset
df = pd.read_csv("store_sales_dataset.csv")

print("Dataset loaded successfully.")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

# View the dataset
print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLast 5 rows:")
print(df.tail().to_string())

print("\nFull dataset:")
print(df.to_string())

print("\nShape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)

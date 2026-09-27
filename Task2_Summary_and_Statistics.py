import pandas as pd

# Load the dataset
df = pd.read_csv("store_sales_dataset.csv")

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Dataset info (columns, non-null counts, dtypes, memory)
print("Dataset info:")
df.info()

# Descriptive statistics for numeric columns
print("\nDescriptive statistics (numeric columns):")
print(df.describe().to_string())

# Descriptive statistics for all columns (including categorical)
print("\nDescriptive statistics (all columns):")
print(df.describe(include="all").to_string())

# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum().to_string())

# Unique values in categorical columns
print("\nUnique values:")
for col in ["Product", "Category", "City"]:
    print(f"{col}: {df[col].nunique()} -> {df[col].unique().tolist()}")

# Extra useful summaries
print("\nTotal sales by category:")
print(df.groupby("Category")["Total_Sales"].sum().to_string())

print("\nTotal sales by city:")
print(df.groupby("City")["Total_Sales"].sum().to_string())

print("\nAverage quantity by product:")
print(df.groupby("Product")["Quantity"].mean().round(2).to_string())

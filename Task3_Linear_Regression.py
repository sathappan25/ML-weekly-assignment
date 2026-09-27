import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 3: Linear Regression for Prediction")
print("=" * 60)

print(f"\nDataset size: {len(df)} rows")
print("\nDataset preview:")
print(df.head().to_string())

# --------------------------------------------------
# 2. Prepare features / target
# X = Unit_Price, y = Total_Sales
# --------------------------------------------------
X = df[["Unit_Price"]]
y = df["Total_Sales"]

print("\nFeatures used: ['Unit_Price']")
print("Target: Total_Sales")

# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# --------------------------------------------------
# 4. Train Linear Regression
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel coefficients:")
print(f"  Unit_Price: {model.coef_[0]:.4f}")
print(f"  Intercept: {model.intercept_:.4f}")

# --------------------------------------------------
# 5. Predict
# --------------------------------------------------
y_pred = model.predict(X_test)

print("\nActual vs Predicted Total_Sales (first 15 test rows):")
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": np.round(y_pred, 2),
})
print(comparison.head(15).to_string(index=False))

# --------------------------------------------------
# 6. Evaluation metrics
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation metrics:")
print(f"  Mean Absolute Error (MAE):  {mae:.2f}")
print(f"  Mean Squared Error (MSE):   {mse:.2f}")
print(f"  Root Mean Squared Error:    {rmse:.2f}")
print(f"  R2 Score:                   {r2:.4f}")

# Sample prediction (Unit_Price only)
sample = pd.DataFrame({"Unit_Price": [20000]})
sample_pred = model.predict(sample)[0]
print("\nSample prediction:")
print("  Unit_Price=20000")
print(f"  Predicted Total_Sales: {sample_pred:.2f}")

# --------------------------------------------------
# 7. Graph: scatter points + ONE straight regression line
# --------------------------------------------------
print("\nOpening graph: Unit_Price vs Total_Sales...")

x_line = np.linspace(
    df["Unit_Price"].min(),
    df["Unit_Price"].max(),
    200,
).reshape(-1, 1)
y_line = model.predict(pd.DataFrame(x_line, columns=["Unit_Price"]))

plt.figure(figsize=(9, 5))
plt.scatter(
    df["Unit_Price"],
    df["Total_Sales"],
    color="steelblue",
    edgecolor="black",
    s=45,
    alpha=0.85,
    label="Data Points",
)
plt.plot(
    x_line,
    y_line,
    color="darkorange",
    linewidth=2,
    label="Regression Line",
)
plt.xlabel("Unit_Price")
plt.ylabel("Total_Sales")
plt.title("Linear Regression: Unit_Price vs Total_Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

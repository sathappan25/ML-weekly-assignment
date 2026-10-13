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

formula_holds = (df["Quantity"] * df["Unit_Price"] == df["Total_Sales"]).all()
print(f"\nTotal_Sales == Quantity x Unit_Price for every row: {formula_holds}")

# --------------------------------------------------
# 2. Prepare features / target
# X = Unit_Price, y = Total_Sales
# --------------------------------------------------
X = df[["Unit_Price"]]
y = df["Total_Sales"]

print("\nFeatures used: ['Unit_Price']")
print("Target: Total_Sales")
print("Note: this is simple (single-feature) linear regression. Quantity is left")
print("out on purpose; section 7 shows how much accuracy that costs.")

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

# --------------------------------------------------
# 7. Limitations of using Unit_Price alone
# --------------------------------------------------
print("\n" + "-" * 60)
print("Limitations of the single-feature model")
print("-" * 60)

print("\nCorrelation with Total_Sales:")
print(df[["Quantity", "Unit_Price", "Total_Sales"]].corr()["Total_Sales"].drop("Total_Sales").round(4).to_string())

# Same train/test rows, but with Quantity added
X2 = df[["Quantity", "Unit_Price"]]
X2_train, X2_test = X2.loc[X_train.index], X2.loc[X_test.index]
model2 = LinearRegression().fit(X2_train, y_train)
y_pred2 = model2.predict(X2_test)
r2_two = r2_score(y_test, y_pred2)

print("\nMultiple linear regression with Quantity + Unit_Price:")
print(f"  Quantity:   {model2.coef_[0]:.4f}")
print(f"  Unit_Price: {model2.coef_[1]:.4f}")
print(f"  Intercept:  {model2.intercept_:.4f}")

metrics = pd.DataFrame({
    "Model": ["Unit_Price only", "Quantity + Unit_Price"],
    "MAE": [mae, mean_absolute_error(y_test, y_pred2)],
    "RMSE": [rmse, np.sqrt(mean_squared_error(y_test, y_pred2))],
    "R2": [r2, r2_two],
})
print("\nTest-set comparison:")
print(metrics.round(4).to_string(index=False))

quantity_band = pd.cut(df.loc[X_test.index, "Quantity"], bins=[0, 10, 20, 30, 40])
error_by_quantity = pd.DataFrame({
    "Orders": quantity_band.value_counts(sort=False),
    "MAE (Unit_Price only)": np.abs(y_test - y_pred).groupby(quantity_band, observed=False).mean(),
})
print("\nError of the single-feature model by Quantity:")
print(error_by_quantity.round(2).to_string())

print("\nWhy the single-feature model is limited:")
print("  - Total_Sales is Quantity x Unit_Price. With Unit_Price alone, orders with")
print("    the same price but different quantities (2 to 40) get the same prediction,")
print(f"    so most of the variation is left unexplained (test R2 = {r2:.4f}).")
print("  - The errors are largest for very small and very large quantities, the orders")
print("    furthest from the average quantity the model implicitly assumes.")
print(f"  - Adding Quantity raises R2 to {r2_two:.4f}, but a linear model adds the")
print("    features (a x Quantity + b x Unit_Price) while the true rule multiplies them,")
print("    so some error remains.")
print("  - A feature equal to Quantity x Unit_Price would give R2 = 1, but that is just")
print("    the formula used to compute Total_Sales, not something the model learns.")

# --------------------------------------------------
# 8. Sample prediction
# --------------------------------------------------
sample = pd.DataFrame({"Unit_Price": [20000]})
sample_pred = model.predict(sample)[0]
sample2 = pd.DataFrame({"Quantity": [10], "Unit_Price": [20000]})
sample_pred2 = model2.predict(sample2)[0]
print("\nSample prediction:")
print("  Unit_Price=20000 (Unit_Price only model)")
print(f"  Predicted Total_Sales: {sample_pred:.2f}")
print("  Quantity=10, Unit_Price=20000 (Quantity + Unit_Price model)")
print(f"  Predicted Total_Sales: {sample_pred2:.2f}")
print(f"  True Total_Sales by the formula: {10 * 20000:.2f}")

# --------------------------------------------------
# 9. Graph: scatter points + ONE straight regression line
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

# --------------------------------------------------
# 10. Graph: actual vs predicted for both models
# --------------------------------------------------
print("\nOpening graph: Actual vs Predicted (single vs two features)...")

limit = [0, max(y_test.max(), y_pred.max(), y_pred2.max()) * 1.05]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
for ax, pred, title, score in [
    (axes[0], y_pred, "Unit_Price only", r2),
    (axes[1], y_pred2, "Quantity + Unit_Price", r2_two),
]:
    ax.scatter(y_test, pred, color="steelblue", edgecolor="black", s=40, alpha=0.8)
    ax.plot(limit, limit, color="darkorange", linewidth=2, label="Perfect prediction")
    ax.set_title(f"{title} (R2 = {score:.3f})")
    ax.set_xlabel("Actual Total_Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Predicted Total_Sales")
plt.tight_layout()
plt.show()

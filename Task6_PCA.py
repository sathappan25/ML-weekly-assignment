import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 6: Principal Component Analysis (PCA)")
print("=" * 60)

print(f"\nDataset size: {len(df)} rows")
print("\nDataset preview:")
print(df.head().to_string())

# --------------------------------------------------
# 2. Prepare features (numerical columns + one-hot City)
# --------------------------------------------------
numeric_features = ["Quantity", "Unit_Price", "Total_Sales"]
X = pd.concat(
    [df[numeric_features], pd.get_dummies(df["City"], prefix="City", dtype=float)],
    axis=1,
)
feature_names = list(X.columns)

print(f"\nFeatures used ({len(feature_names)}):")
for name in feature_names:
    print(f"  {name}")

# --------------------------------------------------
# 3. Standardize the data (PCA is sensitive to scale)
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData standardized (mean = 0, std = 1).")

# --------------------------------------------------
# 4. PCA step by step (from scratch with NumPy)
# --------------------------------------------------
print("\n" + "-" * 60)
print("1. PCA from scratch (Covariance + Eigen decomposition)")
print("-" * 60)

cov_matrix = np.cov(X_scaled, rowvar=False)
print("\nCovariance matrix:")
print(pd.DataFrame(cov_matrix, index=feature_names, columns=feature_names).round(3).to_string())

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# One-hot City columns always sum to 1, so one eigenvalue is ~0 (tiny negatives from rounding)
eigenvalues = np.clip(eigenvalues, 0, None)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

explained_ratio = eigenvalues / eigenvalues.sum()
cumulative_ratio = np.cumsum(explained_ratio)

pc_names = [f"PC{i + 1}" for i in range(len(eigenvalues))]

print("\nEigenvalues and explained variance:")
print(f"  {'PC':<5} {'Eigenvalue':>12} {'Explained':>12} {'Cumulative':>12}")
for name, ev, ratio, cum in zip(pc_names, eigenvalues, explained_ratio, cumulative_ratio):
    print(f"  {name:<5} {ev:>12.4f} {ratio:>12.4f} {cum:>12.4f}")

n_components = int(np.searchsorted(cumulative_ratio, 0.90) + 1)
print(f"\nComponents needed to keep at least 90% variance: {n_components}")

X_pca_manual = X_scaled @ eigenvectors[:, :n_components]

# --------------------------------------------------
# 5. PCA using scikit-learn (verification)
# --------------------------------------------------
print("\n" + "-" * 60)
print("2. PCA using scikit-learn")
print("-" * 60)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance ratio:")
for name, ratio in zip(pc_names, pca.explained_variance_ratio_):
    print(f"  {name}: {ratio:.4f}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.4f}")

# Eigenvector signs are arbitrary, so compare absolute values
match = np.allclose(np.abs(X_pca), np.abs(X_pca_manual), atol=1e-6)
print(f"\nManual PCA matches scikit-learn PCA: {match}")

print("\nPrincipal component loadings (feature contribution):")
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=pc_names[:n_components],
)
print(loadings.round(4).to_string())

print("\nTop contributing feature for each component:")
for pc in loadings.columns:
    top = loadings[pc].abs().idxmax()
    print(f"  {pc}: {top} ({loadings.loc[top, pc]:.4f})")

# --------------------------------------------------
# 6. Transformed data and reconstruction error
# --------------------------------------------------
print(f"\nOriginal shape:    {X_scaled.shape}")
print(f"Transformed shape: {X_pca.shape}")

pca_df = pd.DataFrame(np.round(X_pca, 4), columns=pc_names[:n_components])
pca_df.insert(0, "Category", df["Category"].values)
print("\nTransformed data (first 10 rows):")
print(pca_df.head(10).to_string(index=False))

X_reconstructed = pca.inverse_transform(X_pca)
recon_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction error (MSE, standardized units): {recon_error:.4f}")

# --------------------------------------------------
# 7. Graph: Scree plot (explained variance)
# --------------------------------------------------
print("\nOpening graph: Scree plot...")

x_pos = np.arange(1, len(eigenvalues) + 1)
plt.figure(figsize=(9, 5))
plt.bar(
    x_pos,
    explained_ratio,
    color="steelblue",
    edgecolor="black",
    alpha=0.85,
    label="Individual",
)
plt.step(
    x_pos,
    cumulative_ratio,
    where="mid",
    color="darkorange",
    linewidth=2,
    label="Cumulative",
)
plt.axhline(0.90, color="red", linestyle="--", linewidth=1, label="90% threshold")
plt.xticks(x_pos, pc_names)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Graph: data projected onto PC1 and PC2
# --------------------------------------------------
print("\nOpening graph: Data projected onto PC1 vs PC2...")

pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
styles = {
    "Electronics": {"color": "steelblue", "marker": "o"},
    "Accessories": {"color": "darkorange", "marker": "s"},
}
for category, style in styles.items():
    mask = (df["Category"] == category).values
    plt.scatter(
        X_2d[mask, 0],
        X_2d[mask, 1],
        label=category,
        color=style["color"],
        marker=style["marker"],
        edgecolor="black",
        s=40,
        alpha=0.7,
    )
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.title("PCA: Projection onto First Two Principal Components")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

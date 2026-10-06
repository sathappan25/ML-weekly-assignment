import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 5: K-Means, Gaussian Mixture and Hierarchical Clustering")
print("=" * 60)

print(f"\nDataset size: {len(df)} rows")
print("\nDataset preview:")
print(df.head().to_string())

# --------------------------------------------------
# 2. Prepare features (unsupervised - no target used for training)
# X = Quantity, Unit_Price
# --------------------------------------------------
features = ["Quantity", "Unit_Price"]
X = df[features]

print(f"\nFeatures used for clustering: {features}")
print("Category is NOT used for training; it is only used afterwards")
print("to check how well the clusters match the real categories.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized (mean = 0, std = 1) so both have equal weight.")

# --------------------------------------------------
# 3. Choosing the number of clusters (Elbow + Silhouette)
# --------------------------------------------------
print("\nChoosing number of clusters k (K-Means):")
print(f"  {'k':>3} {'Inertia':>12} {'Silhouette':>12}")

k_values = range(2, 9)
inertias = []
silhouettes = []
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    print(f"  {k:>3} {km.inertia_:>12.2f} {silhouettes[-1]:>12.4f}")

best_k = list(k_values)[int(np.argmax(silhouettes))]
print(f"\nBest k by silhouette score: {best_k}")

print("\nOpening graph: Elbow method and Silhouette scores...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_values), inertias, marker="o", color="steelblue")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster SSE)")
axes[0].set_title("Elbow Method")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_values), silhouettes, marker="o", color="darkorange")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs k")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 4. K-Means clustering
# --------------------------------------------------
print("\n" + "-" * 60)
print(f"1. K-Means Clustering (k = {best_k})")
print("-" * 60)

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster centers (original units):")
print(pd.DataFrame(centers, columns=features).round(2).to_string())

# --------------------------------------------------
# 5. Gaussian Mixture Model (mixture of Gaussians)
# --------------------------------------------------
print("\n" + "-" * 60)
print(f"2. Gaussian Mixture Model (components = {best_k})")
print("-" * 60)

gmm = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
gmm.fit(X_scaled)
gmm_labels = gmm.predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)

print("\nMixture weights:")
for i, w in enumerate(gmm.weights_):
    print(f"  Component {i}: {w:.4f}")

gmm_means = scaler.inverse_transform(gmm.means_)
print("\nComponent means (original units):")
print(pd.DataFrame(gmm_means, columns=features).round(2).to_string())

print(f"\nConverged: {gmm.converged_}")
print(f"Log-likelihood (per sample): {gmm.score(X_scaled):.4f}")
print(f"BIC: {gmm.bic(X_scaled):.2f}")
print(f"AIC: {gmm.aic(X_scaled):.2f}")

print("\nSoft assignments - membership probabilities (first 10 rows):")
prob_df = pd.DataFrame(
    np.round(gmm_probs, 4),
    columns=[f"P(Comp {i})" for i in range(best_k)],
)
prob_df.insert(0, "Unit_Price", df["Unit_Price"].values)
prob_df.insert(0, "Quantity", df["Quantity"].values)
print(prob_df.head(10).to_string(index=False))

# --------------------------------------------------
# 6. Hierarchical (Agglomerative) clustering
# --------------------------------------------------
print("\n" + "-" * 60)
print(f"3. Hierarchical Clustering (Ward linkage, k = {best_k})")
print("-" * 60)

hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
hier_labels = hier.fit_predict(X_scaled)

print("\nOpening graph: Dendrogram...")
linkage_matrix = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, truncate_mode="lastp", p=30, leaf_rotation=90)
plt.xlabel("Sample index (or cluster size)")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Evaluation and comparison
# Silhouette: cluster quality (higher is better, max 1)
# ARI: agreement with real Category (1 = perfect, 0 = random)
# --------------------------------------------------
print("\n" + "=" * 60)
print("Clustering Comparison")
print("=" * 60)

results = {
    "K-Means": kmeans_labels,
    "Gaussian Mixture": gmm_labels,
    "Hierarchical": hier_labels,
}

print(f"\n{'Algorithm':<20} {'Silhouette':>12} {'ARI vs Category':>18}")
for name, labels in results.items():
    sil = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(df["Category"], labels)
    print(f"{name:<20} {sil:>12.4f} {ari:>18.4f}")

for name, labels in results.items():
    print(f"\n{name} - cluster sizes and Category breakdown:")
    print(pd.crosstab(labels, df["Category"], rownames=["Cluster"]).to_string())

# --------------------------------------------------
# 8. Sample prediction (new data point)
# --------------------------------------------------
sample = pd.DataFrame({"Quantity": [10], "Unit_Price": [20000]})
sample_scaled = scaler.transform(sample)

print("\nSample prediction:")
print("  Quantity=10, Unit_Price=20000")
print(f"  K-Means cluster:          {kmeans.predict(sample_scaled)[0]}")
print(f"  Gaussian Mixture cluster: {gmm.predict(sample_scaled)[0]}")
print(f"  GMM probabilities:        {np.round(gmm.predict_proba(sample_scaled)[0], 4)}")

# --------------------------------------------------
# 9. Graph: clusters from all three algorithms
# --------------------------------------------------
print("\nOpening graph: Cluster results of all three algorithms...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, labels) in zip(axes, results.items()):
    ax.scatter(
        df["Unit_Price"],
        df["Quantity"],
        c=labels,
        cmap="viridis",
        edgecolor="black",
        s=35,
        alpha=0.8,
    )
    ax.set_xlabel("Unit_Price")
    ax.set_ylabel("Quantity")
    ax.set_title(name)
    ax.grid(True, alpha=0.3)

axes[0].scatter(
    centers[:, 1],
    centers[:, 0],
    color="red",
    marker="X",
    s=200,
    edgecolor="black",
    label="Centroids",
)
axes[0].legend()
axes[1].scatter(
    gmm_means[:, 1],
    gmm_means[:, 0],
    color="red",
    marker="X",
    s=200,
    edgecolor="black",
    label="Gaussian Means",
)
axes[1].legend()

plt.suptitle("Clustering: Unit_Price vs Quantity")
plt.tight_layout()
plt.show()

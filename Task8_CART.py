import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def gini(labels):
    """Gini impurity: 1 - sum(p_k^2)."""
    p = labels.value_counts(normalize=True)
    return 1.0 - (p ** 2).sum()


def best_split(x, y):
    """Best threshold on one feature by weighted Gini impurity (CART rule)."""
    values = np.sort(x.unique())
    thresholds = (values[:-1] + values[1:]) / 2
    best_threshold, best_gini = None, np.inf
    for thr in thresholds:
        left, right = y[x <= thr], y[x > thr]
        weighted = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
        if weighted < best_gini:
            best_threshold, best_gini = thr, weighted
    return best_threshold, best_gini


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 8: CART Decision Tree for Categorization")
print("=" * 60)

print(f"\nDataset size: {len(df)} rows")
print("\nDataset preview:")
print(df.head().to_string())

# --------------------------------------------------
# 2. Features and target
# Target: Category (Electronics / Accessories)
# Product is excluded because it decides Category exactly.
# Total_Sales is excluded because it is Quantity x Unit_Price.
# --------------------------------------------------
X = pd.concat(
    [df[["Quantity", "Unit_Price"]], pd.get_dummies(df["City"], prefix="City", dtype=int)],
    axis=1,
)
y = df["Category"]
feature_names = list(X.columns)
class_names = sorted(y.unique())

print(f"\nFeatures used ({len(feature_names)}): {feature_names}")
print("\nClass distribution:")
print(y.value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"\nNumber of training samples: {len(X_train)}")
print(f"Number of testing samples:  {len(X_test)}")

# --------------------------------------------------
# 3. How CART chooses a split (Gini impurity, by hand)
# --------------------------------------------------
print("\n" + "-" * 60)
print("1. CART split rule: Gini impurity computed by hand")
print("-" * 60)

root_gini = gini(y_train)
print(f"\nGini impurity of the root node (training data): {root_gini:.4f}")

rows = []
for feature in feature_names:
    thr, g = best_split(X_train[feature], y_train)
    rows.append({"Feature": feature, "Best threshold": thr, "Weighted Gini": g,
                 "Gini decrease": root_gini - g})
split_table = pd.DataFrame(rows).sort_values("Weighted Gini").reset_index(drop=True)
print("\nBest split for each feature:")
print(split_table.round(4).to_string(index=False))

manual_feature = split_table.loc[0, "Feature"]
manual_threshold = split_table.loc[0, "Best threshold"]
print(f"\nBest root split (manual): {manual_feature} <= {manual_threshold:.2f}")

# --------------------------------------------------
# 4. Choosing the tree depth with 5-fold cross-validation
# --------------------------------------------------
print("\n" + "-" * 60)
print("2. Choosing max_depth with 5-fold cross-validation")
print("-" * 60)

depths = range(1, 11)
cv_scores, train_scores = [], []
print(f"\n  {'Depth':>5} {'Train Acc':>10} {'CV Acc':>10}")
for d in depths:
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    cv = cross_val_score(tree, X_train, y_train, cv=5).mean()
    train_acc = tree.fit(X_train, y_train).score(X_train, y_train)
    cv_scores.append(cv)
    train_scores.append(train_acc)
    print(f"  {d:>5} {train_acc:>10.4f} {cv:>10.4f}")

best_depth = list(depths)[int(np.argmax(cv_scores))]
print(f"\nBest max_depth by cross-validation: {best_depth}")

print("\nOpening graph: Accuracy vs tree depth...")
plt.figure(figsize=(9, 5))
plt.plot(list(depths), train_scores, marker="o", color="steelblue", label="Training accuracy")
plt.plot(list(depths), cv_scores, marker="s", color="darkorange", label="Cross-validation accuracy")
plt.axvline(best_depth, color="red", linestyle="--", linewidth=1, label=f"Best depth = {best_depth}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("CART: Accuracy vs Tree Depth")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Final CART model
# --------------------------------------------------
print("\n" + "-" * 60)
print(f"3. Final CART model (Gini, max_depth = {best_depth})")
print("-" * 60)

cart = DecisionTreeClassifier(criterion="gini", max_depth=best_depth, random_state=42)
cart.fit(X_train, y_train)

sk_feature = feature_names[cart.tree_.feature[0]]
sk_threshold = cart.tree_.threshold[0]
print(f"\nRoot split chosen by scikit-learn: {sk_feature} <= {sk_threshold:.2f}")
print(f"Matches the manual Gini calculation: {sk_feature == manual_feature and np.isclose(sk_threshold, manual_threshold)}")

print(f"\nTree depth: {cart.get_depth()}, number of leaves: {cart.get_n_leaves()}")
print("\nDecision rules:")
print(export_text(cart, feature_names=feature_names))

y_pred = cart.predict(X_test)
print(f"Training Accuracy: {cart.score(X_train, y_train):.4f}")
print(f"Testing Accuracy:  {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred, labels=class_names),
    index=class_names,
    columns=class_names,
).to_string())

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print("Predictions (first 10 test rows):")
print(comparison.head(10).to_string(index=False))

importance = pd.Series(cart.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nFeature importance (Gini decrease):")
print(importance.round(4).to_string())

print("\nOpening graph: CART decision tree...")
plt.figure(figsize=(16, 8))
plot_tree(
    cart,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
)
plt.title(f"CART Decision Tree (top levels, max_depth = {best_depth})")
plt.tight_layout()
plt.show()

print("\nOpening graph: Feature importance...")
plt.figure(figsize=(9, 5))
importance.sort_values().plot(kind="barh", color="steelblue", edgecolor="black")
plt.xlabel("Importance")
plt.title("CART Feature Importance")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Sample prediction (new order)
# --------------------------------------------------
sample = pd.DataFrame([{"Quantity": 10, "Unit_Price": 12000, "City": "Chennai"}])
sample_X = pd.concat(
    [sample[["Quantity", "Unit_Price"]], pd.get_dummies(sample["City"], prefix="City", dtype=int)],
    axis=1,
).reindex(columns=feature_names, fill_value=0)

print("\nSample prediction:")
print("  Quantity=10, Unit_Price=12000, City=Chennai")
print(f"  Predicted Category: {cart.predict(sample_X)[0]}")
for name, p in zip(cart.classes_, cart.predict_proba(sample_X)[0]):
    print(f"  P({name}): {p:.4f}")

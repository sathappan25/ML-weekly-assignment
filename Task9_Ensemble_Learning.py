import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 9: Ensemble Learning Models for Classification")
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
# 3. Models
# Single tree = baseline
# Bagging / Random Forest = averaging many trees (variance reduction)
# AdaBoost / Gradient Boosting = boosting weak learners (bias reduction)
# Soft Voting = model averaging of different model types
# --------------------------------------------------
models = {
    "Decision Tree (single)": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42
    ),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "Soft Voting": VotingClassifier(
        estimators=[
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("gb", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
            )),
        ],
        voting="soft",
    ),
}

print("\nModels compared:")
for name in models:
    print(f"  - {name}")

# --------------------------------------------------
# 4. Train and evaluate every model
# --------------------------------------------------
print("\n" + "-" * 60)
print("1. Training and evaluating models")
print("-" * 60)

results = []
predictions = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    results.append({
        "Model": name,
        "Train Acc": model.score(X_train, y_train),
        "CV Acc (mean)": cv.mean(),
        "CV Acc (std)": cv.std(),
        "Test Acc": accuracy_score(y_test, y_pred),
        "Test F1": f1_score(y_test, y_pred, average="weighted"),
    })
    print(f"  {name:<24} CV = {cv.mean():.4f}  Test = {results[-1]['Test Acc']:.4f}")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

best_name = results_df.loc[results_df["CV Acc (mean)"].idxmax(), "Model"]
print(f"\nBest model by cross-validation accuracy: {best_name}")

print("\nOpening graph: Model comparison...")
x = np.arange(len(results_df))
width = 0.38
plt.figure(figsize=(11, 5))
plt.bar(x - width / 2, results_df["CV Acc (mean)"], width,
        yerr=results_df["CV Acc (std)"], capsize=4,
        color="steelblue", edgecolor="black", label="CV accuracy (train)")
plt.bar(x + width / 2, results_df["Test Acc"], width,
        color="darkorange", edgecolor="black", label="Test accuracy")
plt.xticks(x, results_df["Model"], rotation=15)
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.0)
plt.title("Ensemble Models vs Single Decision Tree")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Detailed results for the best model
# --------------------------------------------------
print("\n" + "-" * 60)
print(f"2. Detailed results: {best_name}")
print("-" * 60)

best_pred = predictions[best_name]
print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(
    confusion_matrix(y_test, best_pred, labels=class_names),
    index=class_names,
    columns=class_names,
).to_string())

print("\nClassification Report:")
print(classification_report(y_test, best_pred))

comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": best_pred})
print("Predictions (first 10 test rows):")
print(comparison.head(10).to_string(index=False))

# --------------------------------------------------
# 6. Boosting: choosing the number of estimators with cross-validation
# Only the training data is used to choose; the test set is used once at the end.
# --------------------------------------------------
print("\n" + "-" * 60)
print("3. Boosting: choosing the number of estimators (5-fold CV on training data)")
print("-" * 60)


def staged_cv_curve(model, X, y, cv):
    """Mean validation accuracy after each boosting stage, averaged over the CV folds."""
    curves = []
    for train_idx, val_idx in cv.split(X, y):
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        with warnings.catch_warnings():
            # AdaBoost.staged_predict passes a plain array internally and triggers a spurious feature-name warning
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            curves.append([accuracy_score(y.iloc[val_idx], p) for p in fold_model.staged_predict(X.iloc[val_idx])])
    n_stages = min(len(c) for c in curves)  # AdaBoost can stop early on a fold
    return np.mean([c[:n_stages] for c in curves], axis=0)


cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_curves = {}
tuning = []
for name in ["AdaBoost", "Gradient Boosting"]:
    curve = staged_cv_curve(models[name], X_train, y_train, cv_folds)
    best_n = int(np.argmax(curve)) + 1
    tuned_model = clone(models[name]).set_params(n_estimators=best_n).fit(X_train, y_train)
    cv_curves[name] = (curve, best_n)
    tuning.append({
        "Model": name,
        "Chosen estimators": best_n,
        "CV Acc (chosen)": curve[best_n - 1],
        "Test Acc (200 estimators)": accuracy_score(y_test, predictions[name]),
        "Test Acc (chosen)": accuracy_score(y_test, tuned_model.predict(X_test)),
    })

print(pd.DataFrame(tuning).round(4).to_string(index=False))
print("\nThe number of estimators is picked from the CV curve on the training data only.")
print("The test set is used once, after the choice is made, so the test accuracy stays")
print("an honest estimate of performance on new orders.")

print("\nOpening graph: Boosting CV accuracy vs number of estimators...")
plt.figure(figsize=(9, 5))
for (name, (curve, best_n)), color in zip(cv_curves.items(), ["steelblue", "darkorange"]):
    plt.plot(range(1, len(curve) + 1), curve, color=color, label=name)
    plt.axvline(best_n, color=color, linestyle="--", alpha=0.7,
                label=f"{name}: chosen {best_n}")
plt.xlabel("Number of estimators")
plt.ylabel("Mean CV accuracy (training data)")
plt.title("Boosting: Cross-validated Accuracy vs Number of Estimators")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 7. Random Forest feature importance
# --------------------------------------------------
importance = pd.Series(
    models["Random Forest"].feature_importances_, index=feature_names
).sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(importance.round(4).to_string())

print("\nOpening graph: Random Forest feature importance...")
plt.figure(figsize=(9, 5))
importance.sort_values().plot(kind="barh", color="steelblue", edgecolor="black")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Sample prediction (new order)
# --------------------------------------------------
sample = pd.DataFrame([{"Quantity": 10, "Unit_Price": 12000, "City": "Chennai"}])
sample_X = pd.concat(
    [sample[["Quantity", "Unit_Price"]], pd.get_dummies(sample["City"], prefix="City", dtype=int)],
    axis=1,
).reindex(columns=feature_names, fill_value=0)

print("\nSample prediction:")
print("  Quantity=10, Unit_Price=12000, City=Chennai")
for name, model in models.items():
    proba = model.predict_proba(sample_X)[0]
    probs = ", ".join(f"P({c})={p:.3f}" for c, p in zip(model.classes_, proba))
    print(f"  {name:<24} -> {model.predict(sample_X)[0]:<12} ({probs})")

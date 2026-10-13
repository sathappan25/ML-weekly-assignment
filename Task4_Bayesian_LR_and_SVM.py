import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Check PyMC (required for genuine Bayesian Logistic Regression)
# --------------------------------------------------
try:
    import pymc as pm
    import arviz as az
except ImportError:
    print("PyMC is required for Bayesian Logistic Regression.")
    print("Install it using:")
    print("pip install pymc")
    sys.exit(1)


def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def print_eval_block(name, y_true, y_pred, target_names):
    """Print accuracy, confusion matrix, report, precision, recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{name} Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted):    {recall:.4f}")
    print(f"F1-score (weighted):  {f1:.4f}")
    return acc, cm


def posterior_summary(idata, var_names):
    """az.summary with a 94% interval (ArviZ 1.x uses ci_prob, ArviZ 0.x uses hdi_prob)."""
    try:
        return az.summary(idata, var_names=var_names, ci_prob=0.94)
    except TypeError:
        return az.summary(idata, var_names=var_names, hdi_prob=0.94)


def main():
    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    df = pd.read_csv("store_sales_dataset.csv")

    print("=" * 60)
    print("TASK 4: Bayesian Logistic Regression and SVM Classification")
    print("=" * 60)

    n_rows = df.shape[0]
    print(f"\nDataset size: {n_rows} rows, {df.shape[1]} columns")
    print("\nDataset preview:")
    print(df.head().to_string())

    print(
        "\nTotal_Sales = Quantity x Unit_Price. "
        "It is excluded from the classification features because "
        "it is derived from the other numerical variables."
    )

    # --------------------------------------------------
    # Diagnostics: class balance and feature overlap
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Dataset Diagnostics")
    print("=" * 60)

    print("\nClass distribution:")
    print(df["Category"].value_counts().to_string())
    print("\nClass distribution (proportions):")
    print(df["Category"].value_counts(normalize=True).round(3).to_string())

    print("\nUnit_Price by Category:")
    print(df.groupby("Category")["Unit_Price"].agg(["min", "max", "mean", "median"]).round(2).to_string())

    print("\nQuantity by Category:")
    print(df.groupby("Category")["Quantity"].agg(["min", "max", "mean", "median"]).round(2).to_string())

    acc_prices = df.loc[df["Category"] == "Accessories", "Unit_Price"]
    ele_prices = df.loc[df["Category"] == "Electronics", "Unit_Price"]
    overlap_low = max(acc_prices.min(), ele_prices.min())
    overlap_high = min(acc_prices.max(), ele_prices.max())
    n_acc_overlap = ((acc_prices >= overlap_low) & (acc_prices <= overlap_high)).sum()
    n_ele_overlap = ((ele_prices >= overlap_low) & (ele_prices <= overlap_high)).sum()

    print("\nUnit_Price overlap diagnostic:")
    print(f"  Overlap interval: Rs {overlap_low} to Rs {overlap_high}")
    print(f"  Accessories in overlap: {n_acc_overlap}")
    print(f"  Electronics in overlap: {n_ele_overlap}")
    if overlap_high > overlap_low and n_acc_overlap > 0 and n_ele_overlap > 0:
        print(
            "  Conclusion: Unit_Price ranges OVERLAP between categories, "
            "so Category is not perfectly separable by price alone."
        )
    else:
        print(
            "  Warning: Unit_Price ranges look almost non-overlapping. "
            "Classification may become unrealistically easy."
        )

    print("\nCity vs Category (City should not fully determine Category):")
    print(pd.crosstab(df["City"], df["Category"]).to_string())

    # --------------------------------------------------
    # Visualization: single scatter plot (actual values, no jitter)
    # --------------------------------------------------
    print("\nOpening scatter plot: Unit Price vs Quantity by Category...")
    plt.figure(figsize=(10, 6))
    styles = {
        "Electronics": {"color": "steelblue", "marker": "o"},
        "Accessories": {"color": "darkorange", "marker": "s"},
    }
    for category, style in styles.items():
        part = df[df["Category"] == category]
        plt.scatter(
            part["Unit_Price"],
            part["Quantity"],
            label=category,
            color=style["color"],
            marker=style["marker"],
            s=45,
            alpha=0.6,
        )
    plt.xlabel("Unit Price (₹)")
    plt.ylabel("Quantity")
    plt.title("Unit Price vs Quantity by Category")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # 2. Prepare classification target (binary)
    # --------------------------------------------------
    categories_sorted = sorted(df["Category"].unique())
    category_to_code = {cat: i for i, cat in enumerate(categories_sorted)}
    code_to_category = {i: cat for cat, i in category_to_code.items()}

    y = df["Category"].map(category_to_code).astype(int)

    print("\nCategory mapping (binary target):")
    for cat, code in category_to_code.items():
        print(f"  {cat} -> {code}")

    # --------------------------------------------------
    # 3. Features: Quantity, Unit_Price, City
    # --------------------------------------------------
    X = df[["Quantity", "Unit_Price", "City"]].copy()

    print("\nRaw features used: Quantity, Unit_Price, City")
    print("City will be one-hot encoded (no artificial city ordering).")
    print(
        "The first city is dropped as the reference category, because the "
        "model already has an intercept (keeping every city column would make "
        "the intercept and city effects non-identifiable)."
    )

    # --------------------------------------------------
    # 4. Stratified train/test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"\nNumber of training samples: {len(X_train)}")
    print(f"Number of testing samples:  {len(X_test)}")
    print("\nTrain class distribution:")
    print(y_train.map(code_to_category).value_counts().to_string())
    print("\nTest class distribution:")
    print(y_test.map(code_to_category).value_counts().to_string())

    # --------------------------------------------------
    # 5. Preprocessing (fit on TRAIN only — no leakage)
    # StandardScaler for numeric + OneHotEncoder for City (first city dropped)
    # --------------------------------------------------
    numeric_features = ["Quantity", "Unit_Price"]
    categorical_features = ["City"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    X_train_proc = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    X_test_proc = np.asarray(preprocessor.transform(X_test), dtype=float)

    ohe = preprocessor.named_transformers_["cat"]
    city_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + city_names

    print("\nProcessed feature columns:")
    for name in feature_names:
        print(f"  {name}")
    print(f"  (reference city: {ohe.categories_[0][0]})")

    y_train_np = y_train.to_numpy().astype(int)
    y_test_np = y_test.to_numpy().astype(int)
    n_features = X_train_proc.shape[1]
    target_names = [code_to_category[i] for i in sorted(code_to_category.keys())]

    # --------------------------------------------------
    # 6. Bayesian Logistic Regression with PyMC
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("1. Bayesian Logistic Regression (PyMC)")
    print("-" * 60)
    print(f"Environment: PyMC {pm.__version__}, ArviZ {az.__version__}")
    print("Sampling posterior (this may take a minute)...")

    coords = {"feature": feature_names}

    with pm.Model(coords=coords) as bayesian_logit_model:
        # Priors
        intercept = pm.Normal("intercept", mu=0.0, sigma=2.5)
        betas = pm.Normal("betas", mu=0.0, sigma=2.5, dims="feature")

        # Linear predictor and probability
        logit_p = intercept + pm.math.dot(X_train_proc, betas)
        p = pm.math.sigmoid(logit_p)

        # Bernoulli likelihood
        pm.Bernoulli("y_obs", p=p, observed=y_train_np)

        # cores=1 avoids Windows multiprocessing issues
        # progressbar=False avoids Windows Unicode console errors
        idata = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            cores=1,
            target_accept=0.9,
            random_seed=42,
            progressbar=False,
            return_inferencedata=True,
        )

    print("\nPosterior summary (mean and 94% credible intervals):")
    summary = posterior_summary(idata, ["intercept", "betas"])
    print(summary.to_string())

    r_hat_max = pd.to_numeric(summary["r_hat"], errors="coerce").max()
    ess_min = pd.to_numeric(summary["ess_bulk"], errors="coerce").min()
    n_divergent = int(np.asarray(idata.sample_stats["diverging"]).sum())
    converged = r_hat_max <= 1.01 and ess_min >= 400 and n_divergent == 0

    print("\nConvergence checks:")
    print(f"  Max r_hat:     {r_hat_max:.3f}  (should be <= 1.01)")
    print(f"  Min bulk ESS:  {ess_min:.0f}  (should be >= 400)")
    print(f"  Divergences:   {n_divergent}  (should be 0)")
    if converged:
        print("  Result: the sampler converged, so the posterior can be trusted.")
    else:
        print("  Result: WARNING - the sampler did not clearly converge; do not trust these results yet.")

    # --------------------------------------------------
    # 7. Posterior prediction on test set (genuine averaging)
    # --------------------------------------------------
    intercept_samples = idata.posterior["intercept"].values.reshape(-1)
    beta_samples = idata.posterior["betas"].values.reshape(-1, n_features)

    logit_test = intercept_samples[:, None] + np.dot(beta_samples, X_test_proc.T)
    prob_samples = sigmoid(logit_test)
    prob_mean = prob_samples.mean(axis=0)
    y_pred_blr = (prob_mean >= 0.5).astype(int)

    print("\nBayesian Logistic Regression predictions (first 15 test rows):")
    blr_comparison = pd.DataFrame({
        "Actual": [code_to_category[i] for i in y_test_np],
        "Predicted": [code_to_category[i] for i in y_pred_blr],
        "P(class=1)": np.round(prob_mean, 4),
    })
    print(blr_comparison.head(15).to_string(index=False))

    acc_blr, cm_blr = print_eval_block(
        "Bayesian Logistic Regression", y_test_np, y_pred_blr, target_names
    )

    # --------------------------------------------------
    # 8. SVM (same preprocessed features)
    # --------------------------------------------------
    print("\n" + "-" * 60)
    print("2. Support Vector Machine (SVM)")
    print("-" * 60)

    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_model.fit(X_train_proc, y_train_np)
    y_pred_svm = svm_model.predict(X_test_proc)

    print("\nSVM predictions (first 15 test rows):")
    svm_comparison = pd.DataFrame({
        "Actual": [code_to_category[i] for i in y_test_np],
        "Predicted": [code_to_category[i] for i in y_pred_svm],
    })
    print(svm_comparison.head(15).to_string(index=False))

    acc_svm, cm_svm = print_eval_block("SVM", y_test_np, y_pred_svm, target_names)

    # --------------------------------------------------
    # 9. Fair model comparison
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(f"Bayesian Logistic Regression Accuracy: {acc_blr:.4f}")
    print(f"SVM Accuracy:                          {acc_svm:.4f}")

    # --------------------------------------------------
    # 10. Sample prediction (no Total_Sales)
    # --------------------------------------------------
    sample = pd.DataFrame([{
        "Quantity": 10,
        "Unit_Price": 12000,  # inside the price-overlap region
        "City": "Chennai",
    }])
    sample_proc = np.asarray(preprocessor.transform(sample), dtype=float)

    logit_sample = intercept_samples + np.dot(beta_samples, sample_proc.T).ravel()
    prob_sample_draws = sigmoid(logit_sample)
    prob_sample = float(prob_sample_draws.mean())
    pred_blr_code = int(prob_sample >= 0.5)
    pred_svm_code = int(svm_model.predict(sample_proc)[0])

    print("\n## Sample Prediction")
    print("Quantity: 10")
    print("Unit_Price: 12000")
    print("City: Chennai")
    print("\nBayesian Logistic Regression:")
    print(f"Predicted Category: {code_to_category[pred_blr_code]}")
    print(f"P({code_to_category[1]}): {prob_sample:.4f}")
    print("\nSVM:")
    print(f"Predicted Category: {code_to_category[pred_svm_code]}")


if __name__ == "__main__":
    main()

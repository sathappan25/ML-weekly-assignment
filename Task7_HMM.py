import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# --------------------------------------------------
# HMM building blocks (implemented from scratch with NumPy)
# pi: initial state probabilities      (N,)
# A:  state transition matrix          (N, N)
# B:  emission matrix                  (N, M)
# --------------------------------------------------
PROB_FLOOR = 1e-10  # keeps every probability > 0 so scaling factors never become 0


def normalize(M):
    """Add a small floor and make each row sum to 1."""
    M = np.asarray(M, dtype=float) + PROB_FLOOR
    return M / M.sum(axis=-1, keepdims=True)


def forward(obs, pi, A, B):
    """Scaled forward algorithm. Returns scaled alphas and scaling factors.

    scale[t] = P(o_t | o_1..o_t-1), so log P(sequence) = sum(log(scale)).
    """
    T, N = len(obs), len(pi)
    alpha = np.zeros((T, N))
    scale = np.zeros(T)
    for t in range(T):
        prior = pi if t == 0 else alpha[t - 1] @ A
        alpha[t] = prior * B[:, obs[t]]
        scale[t] = alpha[t].sum()
        if not (np.isfinite(scale[t]) and scale[t] > 0):
            raise FloatingPointError(
                f"Scaling factor is {scale[t]} at t={t}: the model gives this observation zero probability."
            )
        alpha[t] /= scale[t]
    return alpha, scale


def forward_unscaled(obs, pi, A, B):
    """Plain forward algorithm without scaling (only safe for short sequences)."""
    alpha = pi * B[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
    return alpha.sum()


def backward(obs, A, B, scale):
    """Scaled backward algorithm using the forward scaling factors."""
    T, N = len(obs), A.shape[0]
    beta = np.zeros((T, N))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])
        beta[t] /= scale[t + 1]
    return beta


def baum_welch(obs, n_states, n_symbols, n_iter=200, tol=1e-6, seed=0):
    """Maximum likelihood training of an HMM with the EM (Baum-Welch) algorithm."""
    rng = np.random.default_rng(seed)
    pi = normalize(rng.dirichlet(np.ones(n_states)))
    A = normalize(rng.dirichlet(np.ones(n_states), size=n_states))
    B = normalize(rng.dirichlet(np.ones(n_symbols), size=n_states))

    history = []
    for _ in range(n_iter):
        # E-step: sum-product (forward-backward) gives state posteriors
        alpha, scale = forward(obs, pi, A, B)
        beta = backward(obs, A, B, scale)
        history.append(np.log(scale).sum())

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (
            alpha[:-1, :, None]
            * A[None, :, :]
            * (B[:, obs[1:]].T * beta[1:])[:, None, :]
        ) / scale[1:, None, None]

        # M-step: re-estimate parameters (normalize keeps them strictly positive)
        pi = normalize(gamma[0])
        A = normalize(xi.sum(axis=0))
        B = normalize(np.array([gamma[obs == k].sum(axis=0) for k in range(n_symbols)]).T)

        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

    # log-likelihood of the final parameters
    _, scale = forward(obs, pi, A, B)
    history.append(np.log(scale).sum())
    return pi, A, B, history


def viterbi(obs, pi, A, B):
    """Most likely hidden state sequence (computed in log space)."""
    T, N = len(obs), len(pi)
    with np.errstate(divide="ignore"):
        log_pi, log_A, log_B = np.log(pi), np.log(A), np.log(B)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = log_pi + log_B[:, obs[0]]
    for t in range(1, T):
        scores = delta[t - 1][:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores.max(axis=0) + log_B[:, obs[t]]
    path = np.zeros(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path, delta[-1].max()


def next_symbol_probs(alpha_t, A, B):
    """P(next observation | observations so far) from a filtered alpha."""
    return (alpha_t @ A) @ B


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("store_sales_dataset.csv")

print("=" * 60)
print("TASK 7: Hidden Markov Model to Predict Sequential Data")
print("=" * 60)

print(f"\nDataset size: {len(df)} rows")
print("\nDataset preview:")
print(df.head().to_string())

# --------------------------------------------------
# 2. Is Order_ID a valid time order?
# The dataset has no date or time column, so Order_ID is the only ordering.
# --------------------------------------------------
time_columns = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
order_gaps = df["Order_ID"].sort_values().diff().dropna()

print("\nChecking Order_ID as a time order:")
print(f"  Date/time columns in the dataset:  {time_columns if time_columns else 'none'}")
print(f"  Order_ID values are unique:        {df['Order_ID'].is_unique}")
print(f"  Rows already sorted by Order_ID:   {df['Order_ID'].is_monotonic_increasing}")
print(f"  Order_ID consecutive (no gaps):    {(order_gaps == 1).all()} "
      f"({df['Order_ID'].min()} to {df['Order_ID'].max()})")
print("\nAssumption: order numbers are issued one after another, so a higher Order_ID")
print("is a later order. The data is consistent with this (unique, consecutive, sorted)")
print("but cannot prove it, because there is no timestamp. If Order_ID is not the real")
print("time order, the results below describe the row order, not a time series.")

# --------------------------------------------------
# 3. Build the observation sequence
# Orders are taken in Order_ID order (time sequence).
# Each order is labelled Low / Medium / High by Total_Sales.
# --------------------------------------------------
df = df.sort_values("Order_ID").reset_index(drop=True)

level_names = ["Low", "Medium", "High"]
n_symbols = len(level_names)

split = int(len(df) * 0.8)
train_sales = df["Total_Sales"].iloc[:split]
cuts = np.quantile(train_sales, [1 / 3, 2 / 3])
obs = np.digitize(df["Total_Sales"], cuts)

print("\nObserved symbol: sales level of each order (by Total_Sales)")
print(f"  Low    : Total_Sales <  {cuts[0]:,.0f}")
print(f"  Medium : {cuts[0]:,.0f} <= Total_Sales < {cuts[1]:,.0f}")
print(f"  High   : Total_Sales >= {cuts[1]:,.0f}")
print("(Thresholds are the tertiles of the training part only.)")

seq_preview = pd.DataFrame({
    "Order_ID": df["Order_ID"],
    "Total_Sales": df["Total_Sales"],
    "Sales_Level": [level_names[o] for o in obs],
})
print("\nSequence preview (first 10 orders):")
print(seq_preview.head(10).to_string(index=False))

obs_train, obs_test = obs[:split], obs[split:]
print(f"\nTraining sequence length: {len(obs_train)}")
print(f"Testing sequence length:  {len(obs_test)}")

# --------------------------------------------------
# 4. Does the previous order say anything about the next one?
# Chi-square test of independence on consecutive pairs (training part only).
# --------------------------------------------------
pairs = pd.crosstab(
    pd.Categorical([level_names[o] for o in obs_train[:-1]], categories=level_names),
    pd.Categorical([level_names[o] for o in obs_train[1:]], categories=level_names),
    rownames=["Previous"],
    colnames=["Next"],
    dropna=False,
)
chi2, p_value, dof, _ = chi2_contingency(pairs)
lag1 = df["Total_Sales"].iloc[:split].autocorr(lag=1)

print("\nConsecutive sales levels in the training sequence (counts):")
print(pairs.to_string())
print(f"\nChi-square test of independence: chi2 = {chi2:.3f}, dof = {dof}, p-value = {p_value:.4f}")
print(f"Lag-1 autocorrelation of Total_Sales: {lag1:.4f}")
if p_value < 0.05:
    print("The next level depends on the previous one (p < 0.05), so there is")
    print("sequential structure for the HMM to learn.")
else:
    print("No evidence that the next level depends on the previous one (p >= 0.05).")
    print("The HMM can still be fitted, but we should not expect it to beat")
    print("simple baselines at predicting the next order.")

# --------------------------------------------------
# 5. Train the HMM (maximum likelihood with Baum-Welch)
# Several random starts are used because EM can stop at a local optimum.
# --------------------------------------------------
n_states = 2

print("\n" + "-" * 60)
print(f"1. Training HMM ({n_states} hidden states, {n_symbols} symbols) with Baum-Welch")
print("-" * 60)

best = None
for seed in range(5):
    pi, A, B, history = baum_welch(obs_train, n_states, n_symbols, seed=seed)
    print(f"  Start {seed}: log-likelihood = {history[-1]:.4f} after {len(history)} iterations")
    if best is None or history[-1] > best[3][-1]:
        best = (pi, A, B, history)

pi, A, B, history = best

# Order hidden states by their expected sales level so they are easy to read
order = np.argsort(B @ np.arange(n_symbols))
pi, A, B = pi[order], A[np.ix_(order, order)], B[order]
state_names = ["Low-demand state", "High-demand state"]

print(f"\nBest training log-likelihood: {history[-1]:.4f}")
em_steps = np.diff(history)
print(f"Log-likelihood never decreased during EM: {bool((em_steps > -1e-8).all())}")

print("\nInitial state probabilities (pi):")
print(pd.Series(pi, index=state_names).round(4).to_string())

print("\nTransition matrix (A):")
print(pd.DataFrame(A, index=state_names, columns=state_names).round(4).to_string())

print("\nEmission matrix (B):")
print(pd.DataFrame(B, index=state_names, columns=level_names).round(4).to_string())

print("\nOpening graph: Log-likelihood during training...")
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(history) + 1), history, marker="o", markersize=3, color="steelblue")
plt.xlabel("EM iteration")
plt.ylabel("Log-likelihood")
plt.title("Baum-Welch Training: Log-likelihood vs Iteration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Forward algorithm with scaling factors
# scale[t] = P(o_t | o_1..o_t-1), so log P(sequence) = sum of log(scaling factors)
# --------------------------------------------------
print("\n" + "-" * 60)
print("2. Forward algorithm (with scaling factors)")
print("-" * 60)

# Safeguard: scaled and unscaled forward must agree on a short sequence
short = obs_train[:20]
ll_short_scaled = np.log(forward(short, pi, A, B)[1]).sum()
ll_short_plain = np.log(forward_unscaled(short, pi, A, B))
print("Scaling check on the first 20 orders:")
print(f"  log P with scaling:    {ll_short_scaled:.6f}")
print(f"  log P without scaling: {ll_short_plain:.6f}")
print(f"  Match: {np.isclose(ll_short_scaled, ll_short_plain)}")
print(f"Without scaling, P(training sequence) = {forward_unscaled(obs_train, pi, A, B):.3e}")
print(f"(the smallest positive float is about {np.finfo(float).tiny:.1e}, so longer")
print(" sequences would underflow to 0; scaling avoids this)")

# Training: in-sample fit of the 400 training orders
_, scale_train = forward(obs_train, pi, A, B)
ll_train = np.log(scale_train).sum()

# Test: run forward over the whole sequence so every test order is predicted
# from all earlier orders; the test scaling factors give log P(test | training orders)
_, scale_all = forward(obs, pi, A, B)
ll_test = np.log(scale_all[split:]).sum()

# Baseline: orders are independent, with the level frequencies of the training part
level_freq = np.bincount(obs_train, minlength=n_symbols) / len(obs_train)
ll_test_iid = np.log(level_freq[obs_test]).sum()

likelihood = pd.DataFrame({
    "Sequence": ["Training (in-sample)", "Test given training (HMM)", "Test, independent-orders baseline"],
    "Orders": [len(obs_train), len(obs_test), len(obs_test)],
    "Log-likelihood": [ll_train, ll_test, ll_test_iid],
    "Per order": [ll_train / len(obs_train), ll_test / len(obs_test), ll_test_iid / len(obs_test)],
})
likelihood["Perplexity"] = np.exp(-likelihood["Per order"])
print("\nLog-likelihood (higher is better):")
print(likelihood.round(4).to_string(index=False))

print("\nHow to read this:")
print("  - Totals grow with sequence length, so only the per-order values can be compared.")
print("  - The training value is in-sample, so it is optimistic.")
print("  - The test value is log P(test orders | training orders): the forward pass")
print("    continues from the training sequence instead of restarting at pi.")
print("  - Perplexity = exp(-per-order log-likelihood); 3.0 means no better than")
print("    guessing uniformly among the 3 sales levels.")
gain = (ll_test - ll_test_iid) / len(obs_test)
if gain > 0:
    print(f"  - The HMM beats the independent-orders baseline by {gain:.4f} per order,")
    print("    so the hidden states capture some sequential pattern.")
else:
    print(f"  - The HMM is {-gain:.4f} per order worse than the independent-orders")
    print("    baseline, so the hidden states add no predictive value on unseen orders.")

# --------------------------------------------------
# 7. Viterbi decoding (most likely hidden states)
# --------------------------------------------------
print("\n" + "-" * 60)
print("3. Viterbi algorithm (most likely hidden state sequence)")
print("-" * 60)

path, best_log_prob = viterbi(obs, pi, A, B)
print(f"Log-probability of the best state path: {best_log_prob:.4f}")

print("\nTime spent in each hidden state (whole sequence):")
print(pd.Series([state_names[s] for s in path]).value_counts().to_string())

decoded = pd.DataFrame({
    "Order_ID": df["Order_ID"],
    "Sales_Level": [level_names[o] for o in obs],
    "Hidden_State": [state_names[s] for s in path],
})
print("\nDecoded states (first 15 orders):")
print(decoded.head(15).to_string(index=False))

print("\nOpening graph: Observed sales levels and Viterbi hidden states...")
n_show = 100
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].step(df["Order_ID"][:n_show], obs[:n_show], where="mid", color="darkorange")
axes[0].set_yticks(range(n_symbols))
axes[0].set_yticklabels(level_names)
axes[0].set_ylabel("Observed level")
axes[0].set_title(f"Observed Sales Level (first {n_show} orders)")
axes[0].grid(True, alpha=0.3)

axes[1].step(df["Order_ID"][:n_show], path[:n_show], where="mid", color="steelblue")
axes[1].set_yticks(range(n_states))
axes[1].set_yticklabels(state_names)
axes[1].set_xlabel("Order_ID")
axes[1].set_title("Most Likely Hidden State (Viterbi)")
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Predicting the sequence: next order's sales level
# For each test order, only the orders before it are used.
# --------------------------------------------------
print("\n" + "-" * 60)
print("4. Predicting the next sales level (one step ahead)")
print("-" * 60)

alpha_all, _ = forward(obs, pi, A, B)
test_probs = np.array([next_symbol_probs(alpha_all[t - 1], A, B) for t in range(split, len(obs))])
hmm_pred = test_probs.argmax(axis=1)

majority = np.bincount(obs_train, minlength=n_symbols).argmax()
majority_pred = np.full(len(obs_test), majority)
last_pred = obs[split - 1:-1]

pred_df = pd.DataFrame({
    "Order_ID": df["Order_ID"].iloc[split:].values,
    "Actual": [level_names[o] for o in obs_test],
    "Predicted": [level_names[p] for p in hmm_pred],
})
for k, name in enumerate(level_names):
    pred_df[f"P({name})"] = np.round(test_probs[:, k], 4)
print("\nPredictions (first 10 test orders):")
print(pred_df.head(10).to_string(index=False))

hmm_acc = (hmm_pred == obs_test).mean()
majority_acc = (majority_pred == obs_test).mean()
last_acc = (last_pred == obs_test).mean()

print("\nNext-level prediction accuracy on the test sequence:")
print(f"  HMM:                           {hmm_acc:.4f}")
print(f"  Baseline (most frequent level): {majority_acc:.4f}")
print(f"  Baseline (repeat last level):   {last_acc:.4f}")

if hmm_acc > max(majority_acc, last_acc):
    print("\nConclusion: the HMM beats both baselines, so the order sequence")
    print("has a pattern the hidden states can capture.")
else:
    print("\nConclusion: the HMM does not beat the simple baselines. The sales level")
    print("of the next order depends very little on earlier orders, so this")
    print("sequence has only weak sequential structure.")

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(pd.crosstab(
    pd.Categorical([level_names[o] for o in obs_test], categories=level_names),
    pd.Categorical([level_names[p] for p in hmm_pred], categories=level_names),
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False,
).to_string())

# --------------------------------------------------
# 9. Sample prediction (new sequence)
# --------------------------------------------------
sample_levels = ["High", "High", "Medium", "Low", "Low"]
sample_obs = np.array([level_names.index(s) for s in sample_levels])
sample_alpha, _ = forward(sample_obs, pi, A, B)
sample_next = next_symbol_probs(sample_alpha[-1], A, B)
sample_state = state_names[sample_alpha[-1].argmax()]

print("\nSample prediction:")
print(f"  Recent sales levels: {sample_levels}")
print(f"  Current hidden state (most likely): {sample_state}")
for name, p in zip(level_names, sample_next):
    print(f"  P(next = {name}): {p:.4f}")
print(f"  Predicted next sales level: {level_names[sample_next.argmax()]}")

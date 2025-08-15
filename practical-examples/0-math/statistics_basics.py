
"""
Statistics Essentials for Deep Learning
--------------------------------------
Topics:
  - Descriptive stats (mean, std, quantiles)
  - Covariance & correlation
  - Train/validation split with stratification
  - Confidence intervals via bootstrap (no SciPy)
  - Simple A/B test via permutation test
"""

import numpy as np

def descriptive_stats(x):
    x = np.asarray(x)
    stats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
    }
    print("Descriptive stats:", stats)
    return stats

def cov_corr(X):
    """X shape (n_samples, n_features)."""
    X = np.asarray(X)
    C = np.cov(X, rowvar=False)
    R = np.corrcoef(X, rowvar=False)
    print("Covariance:\n", C)
    print("Correlation:\n", R)
    return C, R

def train_val_split(X, y, val_ratio=0.2, seed=0, stratify=False):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = np.arange(n)
    if stratify:
        # simple stratification for binary y {0,1}
        idx0 = idx[y==0]; idx1 = idx[y==1]
        rng.shuffle(idx0); rng.shuffle(idx1)
        n0v = int(len(idx0)*val_ratio); n1v = int(len(idx1)*val_ratio)
        val_idx = np.concatenate([idx0[:n0v], idx1[:n1v]])
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
    else:
        rng.shuffle(idx)
        nval = int(n*val_ratio)
        val_idx, train_idx = idx[:nval], idx[nval:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def bootstrap_ci_mean(x, B=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    means = np.empty(B)
    for b in range(B):
        sample = x[rng.integers(0, n, size=n)]
        means[b] = np.mean(sample)
    lower = float(np.quantile(means, alpha/2))
    upper = float(np.quantile(means, 1-alpha/2))
    print(f"{int((1-alpha)*100)}% CI for mean ≈ [{lower:.4f}, {upper:.4f}]")
    return lower, upper

def permutation_test_ab(a, b, B=5000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    observed = float(np.mean(b) - np.mean(a))
    pooled = np.concatenate([a, b])
    count = 0
    for _ in range(B):
        rng.shuffle(pooled)
        a_perm = pooled[:len(a)]
        b_perm = pooled[len(a):]
        diff = np.mean(b_perm) - np.mean(a_perm)
        if abs(diff) >= abs(observed):
            count += 1
    p_value = (count + 1)/(B + 1)
    print("Permutation test p-value ≈", p_value, " (two-sided)")
    return observed, p_value

def main():
    print("== Statistics Essentials ==")
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    descriptive_stats(x)
    X = rng.normal(size=(200, 3))
    cov_corr(X)
    y = rng.integers(0, 2, size=200)
    Xtr, Xva, ytr, yva = train_val_split(X, y, stratify=True)
    print("Train/Val shapes:", Xtr.shape, Xva.shape)
    bootstrap_ci_mean(x)
    a = rng.normal(loc=0.0, scale=1.0, size=200)
    b = rng.normal(loc=0.2, scale=1.0, size=220)  # small uplift
    permutation_test_ab(a, b)

if __name__ == "__main__":
    main()

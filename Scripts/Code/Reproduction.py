# Necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Data Generation for reproduction
def generate_figure4_data_paper(n=200, p=200, rho=0.8, seed=42):
    np.random.seed(seed)

    # Build covariance matrix Sigma EXACTLY as in the paper
    Sigma = np.eye(p)
    Sigma[0,2] = Sigma[2,0] = rho
    Sigma[1,2] = Sigma[2,1] = rho
    
    # Draw X from N(0, Σ)
    X = np.random.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma,
        size=n
    )

    # True coefficients
    beta = np.zeros(p)
    beta[0] = 1
    beta[1] = 1

    # Noise variance 1/4  →  sd = 1/2  (matching paper)
    eps = 0.5 * np.random.randn(n)

    y = X @ beta + eps

    return X, y, beta

# Standardizing data
def standardize(X):
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1.0
    return X / norms, norms

# Rnadomized Lasso, random weights
def apply_random_weights(X_sub, alpha_weak):
    """
    Two-point Bernoulli randomization:
        W_k = 1        with prob 1/2
        W_k = α_weak   with prob 1/2
    """
    p = X_sub.shape[1]
    W = np.ones(p)
    mask = np.random.rand(p) < 0.5
    W[mask] = alpha_weak
    return X_sub / W[np.newaxis, :]

# Fitting Lasso
def run_lasso_once(X_sub, y_sub, lambda_val):
    n_sub = X_sub.shape[0]
    alpha_sklearn = lambda_val / (2*n_sub)
    model = Lasso(alpha=alpha_sklearn, fit_intercept=False, max_iter=10000)
    model.fit(X_sub, y_sub)
    return (model.coef_ != 0).astype(int)

# Fitting Stabitily path 
def stability_path(Xs, y, lambda_grid, B=100, randomized=False, alpha_weak=1.0):
    n, p = Xs.shape
    m = n // 2   # subsample size
    L = len(lambda_grid)

    selections = np.zeros((B, p, L))

    for b in range(B):
        idx = np.random.choice(n, size=m, replace=False)
        X_sub = Xs[idx]
        y_sub = y[idx]

        if randomized:
            X_sub = apply_random_weights(X_sub, alpha_weak)

        for j, lam in enumerate(lambda_grid):
            selections[b,:,j] = run_lasso_once(X_sub, y_sub, lam)

    return selections.mean(axis=0)

# Making Lamda Grid
def make_lambda_grid(Xs, y, n_lambdas=50):
    lambda_max = np.max(np.abs(Xs.T @ y))
    lambda_min = 0.01 * lambda_max
    lambda_grid = np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambdas)
    return lambda_grid

# Generating the data and Reproducing Figure 4
# 1. Generate data
X, y, beta_true = generate_figure4_data_paper()

# 2. Standardize X
Xs, norms = standardize(X)

# 3. Lambda grid
lambda_grid = make_lambda_grid(Xs, y)

# 4. Stability paths
print("Running α = 1.0 (standard LASSO)...")
Pi_a1 = stability_path(Xs, y, lambda_grid, B=100, randomized=False)

print("Running α = 0.5 (moderate randomization)...")
Pi_a05 = stability_path(Xs, y, lambda_grid, B=100, randomized=True, alpha_weak=0.5)

print("Running α = 0.2 (strong randomization)...")
Pi_a02 = stability_path(Xs, y, lambda_grid, B=100, randomized=True, alpha_weak=0.2)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
paths = [Pi_a1, Pi_a05, Pi_a02]
alphas = [1.0, 0.5, 0.2]
labels = ['(a)', '(b)', '(c)']

for ax, Pi, alpha, lab in zip(axes, paths, alphas, labels):
    
    # TRUE VARIABLES (X1,X2)
    ax.plot(lambda_grid, Pi[0], color='darkred', linewidth=2)
    ax.plot(lambda_grid, Pi[1], color='darkred', linewidth=2)

    # CORRELATED FALSE VAR (X3)
    ax.plot(lambda_grid, Pi[2], color='navy', linewidth=2, linestyle='--')

    # OTHER IRRELEVANT VARS
    for k in range(3, Pi.shape[0]):
        ax.plot(lambda_grid, Pi[k], color='black', alpha=0.3, linewidth=0.3)

    ax.set_xscale('log')
    ax.set_xlim(lambda_grid[-1], lambda_grid[0])
    ax.invert_xaxis()

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("λ", fontsize=12)
    ax.set_ylabel("Π", fontsize=12)
    ax.set_title(f"{lab} α = {alpha}", fontsize=14)

plt.tight_layout()
plt.show()


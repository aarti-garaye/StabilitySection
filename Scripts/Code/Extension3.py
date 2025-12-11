# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


# EXTENDED DATA GENERATION: Random rho
def generate_extended_data(n=200, p=200, n_true=8, n_corr=7,
                           rho_range=(0.5, 0.95),
                           noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)

    # ---- Build covariance matrix ----
    Sigma = np.eye(p)

    # Random diagonal entries (heteroskedastic variances)
    diag_entries = rng.uniform(0.5, 2.0, size=p)
    np.fill_diagonal(Sigma, diag_entries)

    # True variables and correlated irrelevant variables
    true_idx = np.arange(n_true)
    corr_idx = np.arange(n_true, n_true + n_corr)

    # Each correlated variable gets its own rho_j ~ Uniform(rho_range)
    rho_values = rng.uniform(rho_range[0], rho_range[1], size=n_corr)

    for k, j in enumerate(corr_idx):
        t = true_idx[k % n_true]
        rho_j = rho_values[k]
        Sigma[j, t] = Sigma[t, j] = rho_j

    # Make Sigma PSD
    w, v = np.linalg.eigh(Sigma)
    Sigma = v @ np.diag(np.maximum(w, 1e-8)) @ v.T

    # ---- Generate predictors ----
    X = rng.multivariate_normal(np.zeros(p), Sigma, n)

    # Standardize columns
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # ---- True coefficients ----
    beta = np.zeros(p)
    beta[true_idx] = 1

    # ---- Generate response ----
    y = X @ beta + noise_std * rng.normal(size=n)

    return X, y, true_idx, corr_idx, Sigma


# LAMBDA GRID (PAPER CORRECT)
def make_lambda_grid(X, y, n_lambdas=50):
    # PAPER λ_max = max |Xᵀ y|
    lam_max = np.max(np.abs(X.T @ y))
    lam_min = lam_max * 0.01

    # Decreasing grid (1 → 0)
    lam_grid = np.linspace(lam_max, lam_min, n_lambdas)
    return lam_grid, lam_max



# SINGLE STABILITY RUN
def stability_single_run(X, y, subsample_idx, lam,
                         randomized, alpha_weak, rng):
    Xs = X[subsample_idx]
    ys = y[subsample_idx]
    m = len(subsample_idx)

    if randomized:
        # UNIFORM weights (paper correct!)
        W = rng.uniform(alpha_weak, 1.0, X.shape[1])
        Xs = Xs / W

    alpha_sklearn = lam / (2 * m)

    model = Lasso(alpha=alpha_sklearn, fit_intercept=False,
                  max_iter=20000, tol=1e-6)
    model.fit(Xs, ys)

    return (model.coef_ != 0).astype(float)




# Full Stability
def stability_path(X, y, lam_grid, B=200,
                   randomized=False, alpha_weak=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape

    m = n // 2
    L = len(lam_grid)

    sel = np.zeros((p, L))

    for b in range(B):
        subsample = rng.choice(n, m, replace=False)

        for j, lam in enumerate(lam_grid):
            sel[:, j] += stability_single_run(
                X, y, subsample, lam,
                randomized=randomized,
                alpha_weak=alpha_weak,
                rng=rng
            )

        if (b + 1) % 20 == 0:
            print(f"Completed {b+1}/{B}")

    return sel / B



# Plotting
def plot_three_panels(sel1, sel05, sel02, lam_grid,
                      true_idx, corr_idx):

    lam_norm = lam_grid / lam_grid[0]   # normalized to [1 → 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    datasets = [(sel1, 1.0), (sel05, 0.5), (sel02, 0.2)]
    labels = ['(a) α = 1.0', '(b) α = 0.5', '(c) α = 0.2']

    for ax, (sel, alpha), lab in zip(axes, datasets, labels):

        # True vars (red)
        for j in true_idx:
            ax.plot(lam_norm, sel[j], c='darkred', lw=2)

        # Correlated false vars (blue dashed)
        for j in corr_idx:
            ax.plot(lam_norm, sel[j], c='navy', ls='--', lw=2)

        # Noise (faint gray)
        for k in range(sel.shape[0]):
            if k not in true_idx and k not in corr_idx:
                ax.plot(lam_norm, sel[k], c='gray', lw=0.2, alpha=0.2)

        ax.set_xlim(1, 0)
        ax.set_ylim(0, 1.05)
        ax.set_title(lab, fontsize=14)
        ax.set_xlabel("λ", fontsize=12)

    axes[0].set_ylabel("Π", fontsize=12)
    plt.tight_layout()
    plt.show()




# Main
X, y, T, C, Sigma = generate_extended_data(rho_range=(0.5, 0.95),
                                    n_true=8,
                                    n_corr=7,
                                    seed=0)

lam_grid, lam_max = make_lambda_grid(X, y)

sel1  = stability_path(X, y, lam_grid, randomized=False, B=100, seed=1)
sel05 = stability_path(X, y, lam_grid, randomized=True, alpha_weak=0.5, B=100, seed=2)
sel02 = stability_path(X, y, lam_grid, randomized=True, alpha_weak=0.2, B=100, seed=3)

plot_three_panels(sel1, sel05, sel02, lam_grid, T, C)

# Printing correlation matrix
print("Covariance matrix Sigma (first 15x15 block):")
def print_clean_matrix(Sigma, decimals=3, threshold=1e-6):
    Sigma_clean = np.round(Sigma, decimals)
    Sigma_clean[np.abs(Sigma_clean) < threshold] = 0.0
    print(Sigma_clean)
print_clean_matrix(Sigma[:15, :15])
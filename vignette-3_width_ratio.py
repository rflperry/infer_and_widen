# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn import linear_model
from methods import (
    hybrid_inference,
    plausible_LASSO_models_and_signs,
    locally_simultaneous_LASSO,
    simultaneous_PoSI,
)
from utils import powerset, X_in_selected_model
from lasso_utils import (
    lasso_constraints_Xy_space,
)
import matplotlib.pyplot as plt

n_reps = 100
alpha = 0.05

n = 100
p = 10

n_jobs = 10

signal_range = np.asarray([1, 2, 3, 4, 5, 6])
r_range = np.asarray([1, 1.5, 2])

# signal_range = np.asarray([1, 3, 6])
# r_range = np.asarray([1, 2])

nu = 0.1 * alpha

def _single_rep(n, p, rho=0.3, sparsity=0.5, signal_strength=1, r=1, alpha=0.05):
    print(signal_strength, r)
    lam = signal_strength * np.sqrt(2 * np.log(np.e * p))

    beta_star = np.zeros(p)
    beta_star[: np.int_(np.ceil(p * sparsity))] = r * lam

    X = (
        np.sqrt(1 - rho) * np.random.standard_normal((n, p))
        + np.sqrt(rho) * np.random.standard_normal(n)[:, None]
    )
    X /= np.linalg.norm(X, axis=0)
    mu = X @ beta_star
    y = mu + np.random.randn(n)

    # fit LASSO on realized data
    lasso_model = linear_model.Lasso(alpha=lam / n, fit_intercept=False)
    beta_y = lasso_model.fit(X, y).coef_
    M_y = [i for i in range(p) if beta_y[i]]

    if len(M_y) == 0:
        return None

    s_y = np.sign(beta_y[M_y])

    # conditional and hybrid inference
    X_M, X_Mc, E_M, E_Mc = X_in_selected_model(X, M_y)

    # LASSO constraints
    A0_plus, A0_minus, A1, b0_plus, b0_minus, b1 = lasso_constraints_Xy_space(
        lam, X_Mc, X_M, E_M, E_Mc, s_y
    )
    A = np.concatenate((A0_plus, A0_minus, A1))
    b = np.concatenate((b0_plus, b0_minus, b1))

    # simultaneous projection for hybrid
    simultaneous_proj = simultaneous_PoSI(
        X, y, list(powerset(range(p)))[1:], M_y, alpha=nu
    )

    X_M_pinv = np.linalg.pinv(X_M)
    hybrid_ints = [np.zeros(len(M_y)), np.zeros(len(M_y))]

    for k in range(len(M_y)):
        eta = X_M_pinv[k, :]
        SI_truncation = (simultaneous_proj[1][k] - simultaneous_proj[0][k]) / 2
        [hybrid_ints[0][k], hybrid_ints[1][k]] = hybrid_inference(
            y,
            np.eye(n),
            A,
            b,
            eta,
            alpha=alpha / len(M_y),
            beta=nu / len(M_y),
            SI_halfwidth=SI_truncation,
        )

    # locally simultaneous inference
    Ms_pairs = plausible_LASSO_models_and_signs(n, p, lam, X, y, nu, M_y, s_y)
    LSI_ints = locally_simultaneous_LASSO(X, y, Ms_pairs, M_y, alpha=alpha, nu=nu)

    return LSI_ints[1] - LSI_ints[0], hybrid_ints[1] - hybrid_ints[0]


def get_width_ratio(n, p, signal, r, n_reps, rho=0.3, sparsity=0.5, alpha=0.05, n_jobs=1):

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_rep)(n, p, rho, sparsity, signal, r, alpha)
        for _ in tqdm(range(n_reps))
    )

    # results = [_single_rep(mu_vec, Sigma, alpha, nu, beta, plausible_gap) for _ in range(n_reps)]

    iw_widths = np.concat([res[0] for res in results])
    hybrid_widths = np.concat([res[1] for res in results])
    # iw_widths, hybrid_widths = map(np.array, zip(*results))

    return iw_widths.mean() / hybrid_widths.mean()


# Grid
grid = pd.MultiIndex.from_product([signal_range, r_range], names=["signal", "r"]).to_frame(
    index=False
)

# %%
# Compute ratios
grid["ratio"] = [
    get_width_ratio(n=n, p=p, signal=s, r=r, n_reps=n_reps, alpha=alpha, n_jobs=n_jobs)
    for s, r in tqdm(zip(grid["signal"], grid["r"]))
]

# grid["ratio"] = Parallel(n_jobs=10)(
#         delayed(get_width_ratio)(C, n, nu, beta, n_reps, alpha)
#         for C, n in tqdm(zip(grid["C"], grid["n"]))
#     )

#%%
grid.to_csv("data/vignette_3_width_ratios_results.csv", index=False)

# temp_ratio = grid["ratio"].copy()
# grid.loc[grid["ratio"].isna(), "ratio"] = np.inf
# grid.loc[(~temp_ratio.isna()) & (temp_ratio == np.inf), "ratio"] = np.nan
# %%
grid = pd.read_csv("data/vignette_3_width_ratios_results.csv")
heat = grid.pivot(index="r", columns="signal", values="ratio")

from plotting import MidpointNormalize

norm = MidpointNormalize(
    vmin=min(1 - 1e-6, np.nanmin(heat.values)),
    midpoint=1,
    vmax=max(1 + 1e-6, np.nanmax(heat.values)),
)

fig, ax = plt.subplots(figsize=(4.25, 2))

im = ax.imshow(
    heat.values,
    aspect='auto',  # Make tiles rectangles
    cmap="RdBu_r",
    norm=norm,
    origin='lower'
)

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Width Ratio\n(LSI / Hybrid)", fontsize=11)
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_yticks(np.arange(len(heat.index)))
# ax.set_xticklabels([, "", "", f"{heat.columns.values[-1]:.0e}"], fontsize=9)
# ax.set_xticks([0, len(heat.columns) - 1])
ax.set_xticklabels(heat.columns.values, fontsize=9)
ax.set_yticklabels(heat.index.values, fontsize=9)

ax.set_xlabel(r"Selection accuracy (λ₀)", fontsize=11)
ax.set_ylabel(r"Signal (λ/ε)", fontsize=11)

plt.tight_layout()

plt.savefig("figures/vignette_3/vignette-3_width_ratio.png", dpi=300)
plt.close()
# plt.show()
# %%

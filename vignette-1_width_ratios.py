#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from methods import (
    locally_simultaneous_inference,
    hybrid_inference,
    max_z_width,
    plausible_winners,
)
from utils import inference_on_winner_polyhedron
import matplotlib.pyplot as plt


def _single_rep(rep, mu_vec, Sigma, alpha, nu, beta, plausible_gap):
    n = len(mu_vec)

    y = mu_vec + np.random.normal(size=n)
    selected_ind = np.argmax(y)

    # Locally simultaneous (I&W)
    plausible_inds = plausible_winners(y, plausible_gap)
    LSI_int = locally_simultaneous_inference(
        y[selected_ind], Sigma, plausible_inds, [selected_ind],
        alpha=alpha, nu=nu
    )
    iw_width = LSI_int[1][0] - LSI_int[0][0]

    # Hybrid
    A, b = inference_on_winner_polyhedron(n, selected_ind)
    eta = np.zeros(n)
    hybrid_int = hybrid_inference(
        y, Sigma, A, b, eta, alpha=alpha, beta=beta
    )
    hybrid_width = hybrid_int[1] - hybrid_int[0]

    return iw_width, hybrid_width


def get_width_ratio(C, n, n_reps, alpha=0.05, n_jobs=1):
    Sigma = np.eye(n)

    mu_vec = np.zeros(n)
    mu_vec[0] = C * max_z_width(Sigma, alpha)

    nu = 0.1 * alpha
    beta = 0.1 * alpha
    plausible_gap = 4 * max_z_width(np.eye(n), nu)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_single_rep)(
            rep, mu_vec, Sigma, alpha, nu, beta, plausible_gap
        )
        for rep in range(n_reps)
    )

    iw_widths, hybrid_widths = map(np.array, zip(*results))

    return iw_widths.mean() / hybrid_widths.mean()


n_reps = 1
alpha = 0.05

n_range = 10 ** np.arange(1, 6)
C_range = np.asarray([0, 1, 2, 4, 8])

# Grid
grid = pd.MultiIndex.from_product(
    [C_range, n_range],
    names=["C", "n"]
).to_frame(index=False)

#%%
# Compute ratios
grid["ratio"] = [
    get_width_ratio(C, n, n_reps=n_reps, alpha=alpha) for C, n in zip(grid["C"], grid["n"])
]

temp_ratio = grid["ratio"].copy()

# grid.loc[grid["ratio"].isna(), "ratio"] = np.inf
# grid.loc[(~temp_ratio.isna()) & (temp_ratio == np.inf), "ratio"] = np.nan

heat = grid.pivot(index="var", columns="n", values="ratio")

#%%
fig, ax = plt.subplots(figsize=(4, 2))

im = ax.pcolormesh(
    heat.columns.values,
    heat.index.values,
    heat.values,
    shading="auto",
    cmap="Reds"
)

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Width Ratio\n(I&W / S&C)", fontsize=11)

# Log scales
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("n", fontsize=11)
ax.set_ylabel("Winning ", fontsize=11)

# Theme_bw-ish
ax.tick_params(labelsize=11)
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
plt.savefig(
    "figures/vignette_1/vignette-1_width_ratio.png",
    dpi=300
)
plt.close()
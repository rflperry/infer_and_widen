# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from methods import (
    hybrid_inference,
    max_z_width,
    plausible_winners,
)
from utils import inference_on_winner_polyhedron
import matplotlib.pyplot as plt


# MAX_Z_WIDTHS_05 = {
#     10: 3.2994217571329396,
#     100: 3.8864321343742567,
#     1000: 4.420177428265844,
#     10000: 4.565697122149536,
# }

# max_noise = np.amax(np.abs(bstrap_noise), axis=1)

n_reps = 500
alpha = 0.05

n_range = 10 ** np.arange(1, 5)  # 7
r = max_z_width(np.eye(100), alpha)
C_range = np.asarray([0, 1, 2, 4, 6]) * r

num_draws=100000
bstrap_noise = np.random.normal(0, 1, size = (num_draws, max(n_range)))

nu = 0.1 * alpha
beta = 0.1 * alpha

def _single_rep(mu_vec, Sigma, alpha, nu, beta, plausible_gap):
    n = len(mu_vec)

    y = mu_vec + np.random.normal(size=n)
    selected_ind = np.argmax(y)

    # Locally simultaneous (I&W)
    plausible_inds = plausible_winners(y, plausible_gap)
    # Sigma_plausible = Sigma[np.ix_(plausible_inds, plausible_inds)]
    Sigma_selected = Sigma[np.ix_([selected_ind], [selected_ind])]
    # local_halfwidth = np.sqrt(Sigma[selected_ind, selected_ind]) * norm.ppf(1 - ((alpha - nu) / 2 / len(plausible_inds)))
    local_halfwidth = np.quantile(
        np.amax(np.abs(bstrap_noise[:, : len(plausible_inds)]), axis=1),
        1 - (alpha - nu),
    )
    SI_halfwidth = np.quantile(np.amax(np.abs(bstrap_noise[:, :n]), axis=1), 1 - alpha)
    halfwidth = np.minimum(local_halfwidth, SI_halfwidth) * np.sqrt(
        np.diag(Sigma_selected)
    )
    iw_width = 2 * halfwidth

    # LSI_int = locally_simultaneous_inference(
    #     y[selected_ind], Sigma, plausible_inds, [selected_ind],
    #     alpha=alpha, nu=nu
    # )

    # iw_width = LSI_int[1][0] - LSI_int[0][0]

    # Hybrid
    A, b = inference_on_winner_polyhedron(n, selected_ind)
    eta = np.zeros(n)
    eta[selected_ind] = 1
    halfwidth = np.quantile(np.amax(np.abs(bstrap_noise[:, :n]), axis=1), 1 - beta)
    SI_halfwidth = eta.dot(halfwidth * np.sqrt(np.diag(Sigma)))
    hybrid_int = hybrid_inference(y, Sigma, A, b, eta, alpha=alpha, beta=beta, SI_halfwidth=SI_halfwidth)
    hybrid_width = hybrid_int[1] - hybrid_int[0]

    return iw_width, hybrid_width


def get_width_ratio(C, n, nu, beta, n_reps, alpha=0.05, n_jobs=1):
    Sigma = np.eye(n)

    mu_vec = np.zeros(n)
    mu_vec[0] = C

    # results = Parallel(n_jobs=n_jobs)(
    #     delayed(_single_rep)(mu_vec, Sigma, alpha, nu, beta, plausible_gap, bstrap_noise)
    #     for _ in tqdm(range(n_reps))
    # )

    plausible_gap = 4 * np.quantile(np.amax(np.abs(bstrap_noise[:, :n]), axis=1), 1 - nu)

    results = [_single_rep(mu_vec, Sigma, alpha, nu, beta, plausible_gap) for _ in range(n_reps)]

    iw_widths, hybrid_widths = map(np.array, zip(*results))

    return iw_widths.mean() / hybrid_widths.mean()


# Grid
grid = pd.MultiIndex.from_product([C_range, n_range], names=["C", "n"]).to_frame(
    index=False
)

# %%
# Compute ratios
# grid["ratio"] = [
#     get_width_ratio(C, n, nu, beta, n_reps=n_reps, alpha=alpha)
#     for C, n in tqdm(zip(grid["C"], grid["n"]))
# ]

grid["ratio"] = Parallel(n_jobs=10)(
        delayed(get_width_ratio)(C, n, nu, beta, n_reps, alpha)
        for C, n in tqdm(zip(grid["C"], grid["n"]))
    )

grid.to_csv("data/vignette_1_width_ratios_results.csv", index=False)

temp_ratio = grid["ratio"].copy()
grid.loc[grid["ratio"].isna(), "ratio"] = np.inf
grid.loc[(~temp_ratio.isna()) & (temp_ratio == np.inf), "ratio"] = np.nan
# %%
heat = grid.pivot(index="C", columns="n", values="ratio")

# %%
import matplotlib.colors as mcolors

norm = mcolors.TwoSlopeNorm(
    vmin=min(1 - 1e-6, np.nanmin(heat.values)),
    vcenter=1,
    vmax=max(1 + 1e-6, np.nanmax(heat.values)),
)

fig, ax = plt.subplots(figsize=(4.25, 2))

im = ax.imshow(
    heat.values[::-1],
    aspect='auto',  # Make tiles rectangles
    cmap="RdBu_r",
    norm=norm,
)

# Colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Width Ratio\n(LSI / Hybrid)", fontsize=11)
ax.set_xticks(np.arange(len(heat.columns)))
ax.set_yticks(np.arange(len(heat.index)))
# ax.set_xticklabels([, "", "", f"{heat.columns.values[-1]:.0e}"], fontsize=9)
# ax.set_xticks([0, len(heat.columns) - 1])
ax.set_xticklabels([f"{v:.0e}" for v in heat.columns.values], fontsize=9)
ax.set_yticklabels(heat.index.values[::-1].round().astype(int), fontsize=9)

ax.set_xlabel("n", fontsize=11)
ax.set_ylabel("Winner's mean", fontsize=11)

# Theme_bw-ish
ax.tick_params(labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
plt.savefig("figures/vignette_1/vignette-1_width_ratio_temp.png", dpi=300)
plt.close()
# plt.show()
# %%

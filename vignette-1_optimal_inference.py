# %%
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from methods import (
    conditional_inference,
    locally_simultaneous_inference,
    hybrid_inference,
    max_z_width,
    plausible_winners,
    zoom_stepdown,
)
from utils import inference_on_winner_polyhedron
from plotting import plot_oracle


# %%
# one trial of constructing intervals for the winner
# Code modified fromhttps://github.com/tijana-zrnic/locally-simultaneous-inference/blob/main/most-promising-effects.ipynb
def trial(y, Sigma, alpha, nu, beta, plausible_gap, SI_halfwidth):
    n = len(y)
    selected_ind = np.argmax(y)
    scale = np.sqrt(Sigma[selected_ind, selected_ind])

    # naive interval
    half_width = scipy.stats.norm.ppf(1 - alpha / 2) * scale
    naive_int = [y[selected_ind] - half_width, y[selected_ind] + half_width]

    # zoom (step-down)
    zoom_stepdown_int = zoom_stepdown(y, scale, alpha=alpha)

    # simultaneous intervals
    SI_int = [y[selected_ind] - SI_halfwidth, y[selected_ind] + SI_halfwidth]

    # locally simultaneous intervals
    plausible_inds = plausible_winners(y, plausible_gap)
    LSI_int = locally_simultaneous_inference(
        y[selected_ind], Sigma, plausible_inds, [selected_ind], alpha=alpha, nu=nu
    )
    LSI_int = [LSI_int[0][0], LSI_int[1][0]]

    # conditional intervals
    A, b = inference_on_winner_polyhedron(n, selected_ind)
    eta = np.zeros(n)
    eta[selected_ind] = 1
    cond_int = conditional_inference(y, Sigma, A, b, eta, alpha=alpha)

    # hybrid intervals
    hybrid_int = hybrid_inference(y, Sigma, A, b, eta, alpha=alpha, beta=beta)

    return (
        y[selected_ind],
        selected_ind,
        naive_int,
        zoom_stepdown_int,
        SI_int,
        LSI_int,
        cond_int,
        hybrid_int,
    )


# %%
def simulate_rep(mu, rep):
    mu_vec = np.zeros(n)
    mu_vec[0] = mu
    y = mu_vec + np.random.normal(size=n)
    local_results = []
    for alpha in alphas:
        nu = 0.1 * alpha  # for LSI
        beta = 0.1 * alpha  # for hybrid
        plausible_gap = 4 * max_z_width(np.eye(n), nu)
        SI_halfwidth = max_z_width(np.eye(n), alpha)
        try:
            y_sel, sel, *conf_ints = trial(
                y, Sigma, alpha, nu, beta, plausible_gap, SI_halfwidth
            )
        except Exception as e:
            print(e)
            continue

        local_results += [
            rep,
            alpha,
            mu,
            "oracle",
            2 * np.abs(y_sel - mu_vec[sel]),
            np.nan,
        ]

        for method, conf_int in zip(ci_names, conf_ints):
            try:
                covered = int(conf_int[0] <= mu_vec[sel] <= conf_int[1])

                local_results += [
                    rep,
                    alpha,
                    mu,
                    method,
                    conf_int[1] - conf_int[0],
                    covered,
                ]
            except Exception as e:
                print(e)
                local_results += [
                    rep,
                    alpha,
                    mu,
                    method,
                    np.nan,
                    np.nan,
                ]
    return local_results


# %%
# simulation params
n_reps = 1000
alphas = np.array(
    [0.0125, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)
# alphas = np.asarray([0.1, 0.5, 0.9, 0.95])
n = 100
# Cs = [10] # , 30, 50]
# thetas = [0.5, 1, 2, 4]
# ms = np.array([10**j for j in range(1, 4)])

results = []
ci_names = ["naive", "zoom_stepdown", "SI", "LSI", "cond", "hybrid"]

C = 8
Sigma = np.eye(n)
winner_mus = [0, C * max_z_width(Sigma, 0.05)]

np.random.seed(42)
for mu in winner_mus:
    futures = Parallel(n_jobs=-1)(delayed(simulate_rep)(mu, rep) for rep in tqdm(range(n_reps)))
    for res in futures:
        results.extend(res)

df = pd.DataFrame(
    np.array(results).reshape(-1, 6),
    columns=[
        "rep",
        "alpha",
        "mu",
        "method",
        "interval_width",
        "coverage",
    ],
).astype(
    {
        "rep": int,
        "alpha": float,
        "mu": float,
        "method": str,
        "interval_width": float,
        "coverage": float,
    }
)

df.to_csv("data/vignette_1_optimal_inference_results.csv", index=False)

# %%
plot_oracle(df, strat='mu', save_name="figures/vignette_1/vignette-1_oracle-curves.png")
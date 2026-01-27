# %%
#%%
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from methods import (
    wc_laplace_fission,
    wc_alg_stability
)
from plotting import plot_oracle_randomized


# %%
# one trial of constructing intervals for the winner
def trial(y, Sigma, alpha, scale):
    n = len(y)
    
    y_noise = y + np.random.laplace(scale = scale, size = n)
    selected_ind = np.argmax(y_noise)

    sigma = np.sqrt(Sigma[selected_ind, selected_ind])

    # naive interval
    half_width = scipy.stats.norm.ppf(1 - alpha / 2) * sigma
    naive_int = [y[selected_ind] - half_width, y[selected_ind] + half_width]

    # data fission interval
    fission_int = wc_laplace_fission(y[selected_ind], y_noise[selected_ind], scale, alpha, sigma)

    # algorithmic stability interval
    stability_int = wc_alg_stability(y[selected_ind], n, scale, alpha, sigma)

    return (
        y[selected_ind],
        selected_ind,
        naive_int,
        fission_int,
        stability_int,
    )


# %%
def simulate_rep(n, eps, rep):
    mu_vec = np.zeros(n)
    y = np.random.normal(size=n)
    scale = np.sqrt( (1 - eps) / 2 / eps )
    local_results = []
    for alpha in alphas:
        try:
            y_sel, sel, *conf_ints = trial(
                y, Sigma, alpha, scale
            )
        except Exception as e:
            print(e)
            continue

        local_results += [
            rep,
            alpha,
            eps,
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
                    eps,
                    method,
                    conf_int[1] - conf_int[0],
                    covered,
                ]
            except Exception as e:
                print(e)
                local_results += [
                    rep,
                    alpha,
                    eps,
                    method,
                    np.nan,
                    np.nan,
                ]
    return local_results


# %%
# simulation params
n_reps = 1000
alphas = np.array(
    [0.0125, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
)
# alphas = np.asarray([0.1, 0.5, 0.9, 0.95])
n = 100
# Cs = [10] # , 30, 50]
# thetas = [0.5, 1, 2, 4]
# ms = np.array([10**j for j in range(1, 4)])

results = []
ci_names = ["naive", "fission", "stability"]

C = 4
Sigma = np.eye(n)
eps_list = [0.25, 0.75]

np.random.seed(42)
for eps in eps_list:
    futures = Parallel(n_jobs=-1)(delayed(simulate_rep)(n, eps, rep) for rep in tqdm(range(n_reps)))
    for res in futures:
        results.extend(res)

df = pd.DataFrame(
    np.array(results).reshape(-1, 6),
    columns=[
        "rep",
        "alpha",
        "eps",
        "method",
        "interval_width",
        "coverage",
    ],
).astype(
    {
        "rep": int,
        "alpha": float,
        "eps": float,
        "method": str,
        "interval_width": float,
        "coverage": float,
    }
)

df.to_csv("data/vignette_1_optimal_inference_random_results.csv", index=False)

# %%
plot_oracle_randomized(df)
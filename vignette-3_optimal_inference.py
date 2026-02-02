# %%
# %load_ext autoreload
# %autoreload 2
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn import linear_model
from methods import (
    conditional_inference,
    hybrid_inference,
    plausible_LASSO_models_and_signs,
    locally_simultaneous_LASSO,
    simultaneous_PoSI,
)
from utils import powerset, X_in_selected_model
from lasso_utils import (
    lasso_constraints_Xy_space,
)
from plotting import plot_oracle
import os


# %%
# one trial of constructing intervals for the winner
# Code modified fromhttps://github.com/tijana-zrnic/locally-simultaneous-inference/blob/main/lasso.ipynb
def trial(X, y, mu, alpha, nu, lam, target_alpha=0.05):
    n, p = X.shape

    # fit LASSO on realized data
    lasso_model = linear_model.Lasso(alpha=lam / n, fit_intercept=False)
    beta_y = lasso_model.fit(X, y).coef_
    M_y = [i for i in range(p) if beta_y[i]]

    if len(M_y) == 0:
        return None

    s_y = np.sign(beta_y[M_y])

    # projection coefficients for selected variables
    beta_M_star = np.linalg.inv(X[:, M_y].T @ X[:, M_y]) @ X[:, M_y].T @ mu

    # naive intervals
    beta_M = np.linalg.pinv(X[:, M_y]) @ y
    se = np.sqrt(np.diag(np.linalg.inv(X[:, M_y].T @ X[:, M_y])))

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
    cond_ints = [np.zeros(len(M_y)), np.zeros(len(M_y))]
    hybrid_ints = [np.zeros(len(M_y)), np.zeros(len(M_y))]

    for k in range(len(M_y)):
        eta = X_M_pinv[k, :]
        SI_truncation = (simultaneous_proj[1][k] - simultaneous_proj[0][k]) / 2
        [cond_ints[0][k], cond_ints[1][k]] = conditional_inference(
            y, np.eye(n), A, b, eta, alpha=alpha / len(M_y)
        )
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

    if alpha == target_alpha:
        # naive interval
        naive_ints = beta_M + np.asarray(
            [
                -scipy.stats.norm.ppf(1 - alpha / 2) * se,
                scipy.stats.norm.ppf(1 - alpha / 2) * se,
            ]
        )

        # locally simultaneous inference
        Ms_pairs = plausible_LASSO_models_and_signs(n, p, lam, X, y, nu, M_y, s_y)
        LSI_ints = locally_simultaneous_LASSO(X, y, Ms_pairs, M_y, alpha=alpha, nu=nu)

        # fully simultaneous inference
        SI_ints = simultaneous_PoSI(
            X, y, list(powerset(range(p)))[1:], M_y, alpha=alpha
        )

        return (
            beta_M,
            beta_M_star,
            se,
            cond_ints,
            hybrid_ints,
            naive_ints,
            SI_ints,
            LSI_ints,
        )

    return (
        beta_M,
        beta_M_star,
        se,
        cond_ints,
        hybrid_ints,
    )


# %%
def simulate_rep(
    n, p, rep, rho=0.3, sparsity=0.5, signal_strength=1, r=1.5, target_alpha=0.05
):
    # lam = signal_strength * np.sqrt(2 * np.log(p) * n)
    lam = signal_strength * np.sqrt(2 * np.log(np.e * p))

    beta_star = np.zeros(p)
    beta_star[: np.int_(np.ceil(p * sparsity))] = r * lam
    # beta_star[:np.int_(np.ceil(p * sparsity))] = np.linspace(0, 1, np.int_(np.ceil(p * sparsity))+1)[1:]

    X = (
        np.sqrt(1 - rho) * np.random.standard_normal((n, p))
        + np.sqrt(rho) * np.random.standard_normal(n)[:, None]
    )
    X /= np.linalg.norm(X, axis=0)
    mu = X @ beta_star
    y = mu + np.random.randn(n)

    local_results = []
    for alpha in alphas:
        nu = 0.1 * alpha  # for LSI and hybrid
        try:
            beta_M, beta_M_star, se, *conf_ints = trial(
                X, y, mu, alpha, nu, lam, target_alpha=target_alpha
            )
        except Exception as e:
            print(e)
            continue

        # here, we scale the bias by the sandwich standard error as would be done
        # in the naive setting
        for se_i, bias_i in zip(se, beta_M - beta_M_star):
            local_results += [
                rep,
                alpha,
                rho,
                sparsity,
                signal_strength,
                r,
                "oracle",
                2 * np.abs(bias_i / se_i),
                se_i,
                np.nan,
            ]

        for method, conf_int in zip(ci_names, conf_ints):
            for se_i, beta_star_i, conf_int_i in zip(
                se, beta_M_star, np.asarray(conf_int).T
            ):
                try:
                    covered = int(conf_int_i[0] <= beta_star_i <= conf_int_i[1])
                    local_results += [
                        rep,
                        alpha,
                        rho,
                        sparsity,
                        signal_strength,
                        r,
                        method,
                        conf_int_i[1] - conf_int_i[0],
                        se_i,
                        covered,
                    ]
                except Exception as e:
                    print(e)
                    local_results += [
                        rep,
                        alpha,
                        rho,
                        sparsity,
                        signal_strength,
                        r,
                        method,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
    return local_results


# %%
# simulation params
n_reps = 100
alphas = np.array(
    [0.0125, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)
# alphas = np.asarray([0.05, 0.5, 0.90])
target_alpha = 0.05
rho = 0.3
sparsity = 0.5
n = 100
p = 10
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))  # 8

results = []
columns = [
    "rep",
    "alpha",
    "rho",
    "sparsity",
    "signal_strength",
    "r",
    "method",
    "interval_width",
    "se",
    "coverage",
]
ci_names = ["cond", "hybrid", "naive", "SI", "LSI"]

Sigma = np.eye(n)
signal_strength = 1.5 # [0.02, 0.002]
r_values = [1, 4]  # 1

np.random.seed(42)
for r in r_values:
    futures = Parallel(n_jobs=n_jobs)(
        delayed(simulate_rep)(
            n,
            p,
            rep,
            r=r,
            rho=rho,
            signal_strength=signal_strength,
            sparsity=sparsity,
            target_alpha=target_alpha,
        )
        for rep in tqdm(range(n_reps))
    )
    for res in futures:
        results.extend(res)

df = pd.DataFrame(np.array(results).reshape(-1, len(columns)), columns=columns).astype(
    {
        "rep": int,
        "alpha": float,
        "rho": float,
        "sparsity": float,
        "signal_strength": float,
        "r": float,
        "method": str,
        "interval_width": float,
        "se": float,
        "coverage": float,
    }
)

# %%
df.to_csv("data/vignette_3_optimal_inference_results.csv", index=False)

# %%
plot_oracle(
    df,
    strat="r",
    scale=True,
    save_name="figures/vignette_3/vignette-3_oracle_curves.png",
)

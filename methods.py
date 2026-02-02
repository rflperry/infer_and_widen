import numpy as np
from scipy.optimize import brentq
from scipy.stats import truncnorm, norm
from utils import max_z_width, max_Xz
from lasso_utils import all_LASSO_models_and_signs_in_box

"""
Conditional inference
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def conditional_inference(
    y, Sigma, A, b, eta, alpha=0.1, grid_radius=10, num_gridpoints=10000
):

    m = len(y)

    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c, eta)) @ y
    Az = A @ z
    Ac = A @ c
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = -np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf

    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)

    grid = np.linspace(eta_dot_y - grid_radius, eta_dot_y + grid_radius, num_gridpoints)

    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False

    for mu in grid:
        num = norm.cdf((eta_dot_y - mu) / sigma) - norm.cdf(
            (V_minus - mu) / sigma
        )
        denom = norm.cdf((V_plus - mu) / sigma) - norm.cdf(
            (V_minus - mu) / sigma
        )

        if not found_l:
            if num / denom < 1 - alpha / 2:
                ci_l = mu
                found_l = True

        if not found_u and mu >= eta_dot_y:
            if num / denom < alpha / 2:
                ci_u = mu
                found_u = True

        if found_u and found_l:
            break

    return [ci_l, ci_u]


"""
Hybrid inference
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def hybrid_inference(
    y, Sigma, A, b, eta, alpha=0.1, beta=0.01, num_gridpoints=10000, SI_halfwidth=None
):

    m = len(y)

    c = Sigma @ eta / (eta.T @ Sigma @ eta)
    z = (np.eye(m) - np.outer(c, eta)) @ y
    Az = A @ z
    Ac = A @ c
    V_frac = np.divide(b - Az, Ac)
    if (Ac < 0).any():
        V_minus = np.max(V_frac[Ac < 0])
    else:
        V_minus = -np.inf
    if (Ac > 0).any():
        V_plus = np.min(V_frac[Ac > 0])
    else:
        V_plus = np.inf

    eta_dot_y = eta.dot(y)
    sigma = np.sqrt(eta.T @ Sigma @ eta)

    if SI_halfwidth == None:
        SI_halfwidth = eta.dot(max_z_width(Sigma, beta) * np.sqrt(np.diag(Sigma)))

    grid = np.linspace(
        eta_dot_y - SI_halfwidth, eta_dot_y + SI_halfwidth, num_gridpoints
    )

    ci_l = grid[0]
    ci_u = grid[-1]
    found_l = False
    found_u = False

    for i, mu in enumerate(grid):
        V_minus_hybrid = np.maximum(V_minus, mu - SI_halfwidth)
        V_plus_hybrid = np.minimum(V_plus, mu + SI_halfwidth)

        num = norm.cdf((eta_dot_y - mu) / sigma) - norm.cdf(
            (V_minus_hybrid - mu) / sigma
        )
        denom = norm.cdf(
            (V_plus_hybrid - mu) / sigma
        ) - norm.cdf((V_minus_hybrid - mu) / sigma)
        if not found_l:
            if num / denom < 1 - (alpha - beta) / (2 * (1 - beta)):
                ci_l = mu
                found_l = True

        if not found_u and mu >= eta_dot_y:
            if num / denom < (alpha - beta) / (2 * (1 - beta)):
                ci_u = mu
                found_u = True

        if found_u and found_l:
            break

    return [ci_l, ci_u]


"""
Max-z simultaneous inference
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def max_z_inference(point_estimate, Sigma, alpha=0.1):
    halfwidth = max_z_width(Sigma, alpha) * np.sqrt(np.diag(Sigma))
    return [point_estimate - halfwidth, point_estimate + halfwidth]


"""
Locally simultaneous inference
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def locally_simultaneous_inference(
    point_estimate, Sigma, plausible_inds, selected_inds, alpha=0.1, nu=0.01
):
    Sigma_plausible = Sigma[np.ix_(plausible_inds, plausible_inds)]
    Sigma_selected = Sigma[np.ix_(selected_inds, selected_inds)]
    local_halfwidth = max_z_width(Sigma_plausible, alpha - nu)
    SI_halfwidth = max_z_width(Sigma, alpha)
    halfwidth = np.minimum(local_halfwidth, SI_halfwidth) * np.sqrt(
        np.diag(Sigma_selected)
    )
    return [point_estimate - halfwidth, point_estimate + halfwidth]


def plausible_winners(y, plausible_gap):
    return np.where(y >= np.max(y) - plausible_gap)[0]


def plausible_filedrawer(y, plausible_gap, T):
    return np.where(y + plausible_gap > T)[0]


"""
Fully simultaneous PoSI
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def simultaneous_PoSI(X, y, model_space, M_y, alpha=0.1, num_draws=10000):
    curr_max_t = np.zeros(num_draws)
    n, d = X.shape
    bstrap_samples = np.random.multivariate_normal(np.zeros(n), np.eye(n), num_draws)
    for i, Mhat in enumerate(model_space):
        X_Mhat = X[:, Mhat]
        Sigma_inv = np.linalg.inv(X_Mhat.T @ X_Mhat)
        normalizing_vec = np.reshape(np.sqrt(np.diag(Sigma_inv)), (-1, 1))
        t_stats = np.divide(
            np.abs(Sigma_inv @ X_Mhat.T @ bstrap_samples.T), normalizing_vec
        )
        max_t_stat_in_M = np.amax(t_stats, axis=0)
        curr_max_t = np.maximum(max_t_stat_in_M, curr_max_t)
    PoSI_constant = np.quantile(curr_max_t, 1 - alpha)
    X_Mhat = X[:, M_y]
    pointest = np.linalg.pinv(X_Mhat) @ y
    Sigma_inv = np.linalg.inv(X_Mhat.T @ X_Mhat)
    return [
        pointest - PoSI_constant * np.sqrt(np.diag(Sigma_inv)),
        pointest + PoSI_constant * np.sqrt(np.diag(Sigma_inv)),
    ]


"""
Locally simultaneous inference for the LASSO
Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
"""


def plausible_LASSO_models_and_signs(n, d, lam, X, y, nu, M_y, s_y):
    # compute box width
    s_nu = 2 * max_Xz(X, err_level=nu)

    # find all models and signs in box
    M_s_done = all_LASSO_models_and_signs_in_box(X, y, lam, s_nu, M_y, s_y)

    return M_s_done


def locally_simultaneous_LASSO(X, y, M_s_done, M_y, alpha=0.1, nu=0.01):

    # set of models (no signs)
    models = [M_s_done[i][0] for i in range(len(M_s_done)) if M_s_done[i][0]]

    # constructing intervals with local correction
    LSI_int = simultaneous_PoSI(X, y, models, M_y, alpha=(alpha - nu))
    return LSI_int


# Zoom correction (grid search)
# Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
def zoom_grid(X, Sigma, alpha=0.1, simulation_draws=10000, grid_points=1000):
    max_radius = max_z_width(Sigma, alpha) * np.sqrt(np.max(np.diag(Sigma)))
    ihat = np.argmax(X)
    l = X[ihat] - max_radius
    u = X[ihat] + max_radius
    noise_mat = np.abs(
        np.random.multivariate_normal(np.zeros(len(X)), Sigma, simulation_draws)
    )
    # lower bound
    for t in np.linspace(X[ihat] - max_radius, X[ihat], grid_points):
        theta_t = np.minimum(2 / 3 * X + 1 / 3 * t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        noise_mat_max = np.max(
            noise_mat * np.greater(noise_mat, Deltas_t.T / 2), axis=1
        )
        radius = np.quantile(noise_mat_max, 1 - alpha)  # active radius
        if radius > X[ihat] - t:
            l = t
            break
    # upper bound
    for t in np.linspace(X[ihat], X[ihat] + max_radius, grid_points):
        theta_t = np.minimum(2 / 3 * X + 1 / 3 * t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        noise_mat_max = np.max(
            noise_mat * np.greater(noise_mat, Deltas_t.T / 2), axis=1
        )
        radius = np.quantile(noise_mat_max, 1 - alpha)  # active radius
        if radius < t - X[ihat]:
            u = t
            break
    return [l, u]


# Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
def zoom_union_bound(X, sigmas, alpha=0.1, grid_points=1000):
    m = len(X)
    max_radius = norm.isf(alpha / (2 * m)) * np.max(sigmas)
    ihat = np.argmax(X)
    radius_grid = np.linspace(0, max_radius, grid_points)
    # lower bound
    for t in np.linspace(X[ihat] - max_radius, X[ihat], grid_points):
        theta_t = np.minimum(2 / 3 * X + 1 / 3 * t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        tail_vals = np.array(
            [
                tail_bound(radius_grid[j], Deltas_t, sigmas)
                for j in range(len(radius_grid))
            ]
        )
        radius = radius_grid[np.where(tail_vals <= alpha)[0][0]]  # active radius
        if radius > X[ihat] - t:
            l = t
            break
    # upper bound
    for t in np.linspace(X[ihat], X[ihat] + max_radius, grid_points):
        theta_t = np.minimum(2 / 3 * X + 1 / 3 * t, t)
        theta_t[ihat] = t
        Deltas_t = t - theta_t
        tail_vals = np.array(
            [
                tail_bound(radius_grid[j], Deltas_t, sigmas)
                for j in range(len(radius_grid))
            ]
        )
        radius = radius_grid[np.where(tail_vals <= alpha)[0][0]]  # active radius
        if radius < t - X[ihat]:
            u = t
            break
    return [l, u]


# Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
def tail_bound(r, Deltas, sigmas):
    return np.sum(
        [
            2 * norm.sf(np.maximum(r, Deltas[j] / 2), scale=sigmas[j])
            for j in range(len(Deltas))
        ]
    )


# Zoom correction (step-down)
# Code acquired from https://github.com/tijana-zrnic/winners-curse/blob/main/methods.py
def zoom_stepdown(X, sigma, alpha=0.1):
    Deltas = -np.sort(X - np.max(X))
    m = len(Deltas)
    # lower bound
    alpha_hat = alpha
    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2 * (m - k)), scale=sigma)
        if Deltas[k] <= 4 * r_hat_k:
            r_l = r_hat_k
            break
        else:
            alpha_hat -= 2 * norm.sf((Deltas[k] - r_hat_k) / 3, scale=sigma)
    # upper bound
    alpha_hat = alpha
    for k in range(m):
        r_hat_k = norm.isf(alpha_hat / (2 * (m - k)), scale=sigma)
        if Deltas[k] <= 2 * r_hat_k:
            r_u = r_hat_k
            break
        else:
            alpha_hat -= 2 * norm.sf(
                (Deltas[k] + norm.isf(alpha / 2, scale=sigma)) / 3, scale=sigma
            )
    return [np.max(X) - r_l, np.max(X) + r_u]


# Winner's curse: data fission with laplace noise
def wc_laplace_fission(y, y_tr, scale = 1, alpha=0.05, sigma=1, lb=-10, ub=10):    
    lower_tail = (y_tr < y)

    def trunc_cdf(mu):
        if lower_tail:
            a_std = (y_tr - mu) / sigma
            b_std = np.inf
        else:
            a_std = -np.inf
            b_std = (y_tr - mu) / sigma
        return truncnorm.cdf(y, a_std, b_std, loc=mu, scale=sigma)
    
    def solve_mu(p):
        return brentq(lambda mu: trunc_cdf(mu) - p, lb, ub)

    # Invert CDF
    b = solve_mu(alpha / 2)
    a = solve_mu(1 - alpha / 2)

    sign = 1 if lower_tail else -1
    a += sign * sigma**2 / scale
    b += sign * sigma**2 / scale
    
    return np.asarray([a, b])

# Winner's curse: algorithmic stability with laplace noise
def wc_alg_stability(y_tr, n, scale = 1, alpha=0.05, sigma=1):    
    nu = np.linspace(0, alpha, 20)

    h = 2 * norm.ppf(1 - nu / (2 * n))
    eta = h / scale

    as_width = sigma * norm.ppf(1 - (alpha - nu) * np.exp(-eta) / 2)
    as_width = as_width.min()
    
    return y_tr + np.asarray([-as_width, as_width])

# sandiwich covariance inference for OLS
def sandwich_ci(X, y, alpha=0.05):
    """
    X: (n, p) design matrix (include intercept if desired)
    y: (n,) response
    beta_hat: (p,) estimated coefficients
    """

    # Mean estimation
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y

    # Sandwich covariance (HC0)
    resid = y - X @ beta_hat
    S = np.diag(resid**2)
    V = XtX_inv @ X.T @ S @ X @ XtX_inv
    se = np.sqrt(np.diag(V))
    z = norm.ppf(1 - alpha / 2)

    return beta_hat + np.asarray([-z * se, z * se]), se
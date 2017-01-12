import numpy as np
import scipy.stats.distributions as sc_dist


def aggarwal_limits(mu, alpha=0.68268949):
    if isinstance(alpha, float):
        alpha = [alpha]
    mu_large = np.zeros((len(mu), len(alpha)))
    alpha_large = np.zeros_like(mu_large)
    for i, a_i in enumerate(alpha):
        alpha_large[:, i] = a_i
        mu_large[:, i] = mu

    mu_large_flat = mu_large.reshape(np.prod(mu_large.shape))
    alpha_large_flat = alpha_large.reshape(mu_large_flat.shape)
    lower, upper = sc_dist.poisson.interval(alpha_large_flat, mu_large_flat)
    return lower.reshape(mu_large.shape), upper.reshape(mu_large.shape)


def map_aggarwal_ratio(y_values, y_min=None, y_0=1.):
    flattened_y = y_values.reshape(np.prod(y_values.shape))
    pos_infinite = np.isposinf(flattened_y)
    neg_infinite = np.isneginf(flattened_y)
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    finite_y[finite_y > y_0] = np.NaN
    plus_mask = finite_y > 0
    minus_mask = finite_y < 0
    finite_y = np.absolute(finite_y)
    finite_y[plus_mask] = np.log10(finite_y[plus_mask])
    finite_y[minus_mask] = np.log10(finite_y[minus_mask])
    if y_min is None:
        try:
            y_min_plus = np.min(finite_y[plus_mask])
        except ValueError:
            y_min_plus = np.inf
        try:
            y_min_minus = np.min(finite_y[minus_mask])
        except ValueError:
            y_min_minus = np.inf
        y_min = min(y_min_plus, y_min_minus)
        if np.isinf(y_min):
            raise ValueError
        y_min *= 1.1
    finite_y /= np.absolute(y_min)
    finite_y[finite_y > 1] = 1.1
    finite_y[finite_y < -1] = -1.1
    finite_y[minus_mask] *= -1
    tranformed_values = np.zeros_like(flattened_y)
    tranformed_values[:] = np.NaN
    tranformed_values[finite] = finite_y
    tranformed_values[pos_infinite] = flattened_y[pos_infinite]
    tranformed_values[neg_infinite] = flattened_y[neg_infinite]
    tranformed_values = tranformed_values.reshape(y_values.shape) * -1
    return tranformed_values, y_0, y_min


def calc_p_alpha_limits(mu, rel_std):
    a_ref = sc_dist.poisson.cdf(mu, mu)
    abs_std = np.zeros_like(rel_std)
    for i in range(rel_std.shape[1]):
        for j in range(rel_std.shape[2]):
            abs_std[:, i, j] = mu * rel_std[:, i, j]
    lower = abs_std[:, :, 0]
    upper = abs_std[:, :, 1]

    limits = np.ones_like(abs_std)
    limits_lower = limits[:, :, 0]
    limits_upper = limits[:, :, 1]

    for i, [mu_i, a_i] in enumerate(zip(mu, a_ref)):
        if mu_i > 0:
            a_shape = limits_lower[i].shape
            x_lower = lower[i].reshape(np.prod(a_shape))
            x_upper = upper[i].reshape(np.prod(a_shape))
            a_lower = sc_dist.poisson.cdf(x_lower, mu_i)
            a_upper = sc_dist.poisson.cdf(x_upper, mu_i)
            a_lower -= sc_dist.poisson.pmf(x_lower, mu_i)
            a_upper = (1 - a_upper)
            a_lower = (a_lower)
            a_upper /= (1 - a_i)
            a_lower /= a_i
            a_lower[x_lower == 0] = -np.inf
            a_upper[x_upper == 0] = np.inf
            limits_lower[i] = a_lower.reshape(a_shape)
            limits_upper[i] = a_upper.reshape(a_shape)
        else:
            a_shape = limits_lower[i].shape
            x_lower = lower[i].reshape(np.prod(a_shape))
            x_upper = upper[i].reshape(np.prod(a_shape))
            a_lower = np.zeros_like(x_upper) * np.nan
            a_upper = np.zeros_like(x_upper) * np.nan
            a_lower[x_lower > 0] = -np.inf
            a_upper[x_upper > 0] = np.inf
            limits_lower[i] = a_lower.reshape(a_shape)
            limits_upper[i] = a_upper.reshape(a_shape)
    limits[:, :, 0] = limits_lower * -1
    limits[:, :, 1] = limits_upper
    return limits


def calc_p_alpha_ratio(mu, k):
    a_ref = sc_dist.poisson.cdf(mu, mu)
    ratio = np.empty_like(k)
    for i, [mu_i, a_i, k_i] in enumerate(zip(mu, a_ref, k)):
        a = sc_dist.poisson.cdf(k_i, mu_i)
        if k_i == 0 and mu_i == 0:
            ratio[i] = np.NaN
        elif mu_i == 0:
            ratio[i] = np.inf
        elif k_i == 0:
            ratio[i] = -np.inf
        elif k_i >= mu_i:
            if (1 - a_i) == 0:
                ratio[i] = np.inf
            else:
                ratio[i] = (1 - a) / (1 - a_i)
        else:
            a_0 = sc_dist.poisson.pmf(k_i, mu_i)
            ratio[i] = (a - a_0) / (-1 * a_i)
            if (a - a_0) == 0:
                ratio[i] = -np.inf
            else:
                ratio[i] = (a - a_0) / (-1 * a_i)
    return ratio

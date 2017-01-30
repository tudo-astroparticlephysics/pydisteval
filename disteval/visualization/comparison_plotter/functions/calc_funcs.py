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


def map_aggarwal_ratio(y_values, y_min=None, y_0=1., upper=True):
    flattened_y = y_values.reshape(np.prod(y_values.shape))
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    finite_y[finite_y > y_0] = np.NaN
    finite_y = np.log10(finite_y)
    y_min = np.min(finite_y)
    y_min *= 1.1
    finite_y /= y_min
    transformed_values = np.zeros_like(flattened_y)
    transformed_values[finite] = finite_y
    return transformed_values.reshape(y_values.shape), y_min


def rescale_mapping(values, y_min, y_min_wanted):
    finite = np.isfinite(values)
    factor = y_min_wanted / y_min
    values[finite] *= factor
    return values


def calc_p_alpha_limits(mu, rel_std, upper=True):
    abs_std = np.zeros_like(rel_std)
    limits = np.zeros_like(rel_std)
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i]
        limits[:, i] = __calc_p_alpha__(mu, abs_std, upper=upper)
    return limits


def __calc_p_alpha__(mu, k, upper=True):
    assert mu.shape == k.shape, 'Shape of \'mu\' and \'k\' have to be the same'
    limit = np.zeros_like(k)
    is_zero_mu = np.isclose(mu, 0.)
    is_zero_k = np.isclose(k, 0.)
    both_zero = np.logical_and(is_zero_k, is_zero_mu)
    is_finite = ~is_zero_mu

    limit[both_zero] = np.NaN
    limit[is_zero_mu] = -np.inf
    a_ref = sc_dist.poisson.cdf(mu[is_finite], mu[is_finite])
    a_k = sc_dist.poisson.cdf(k[is_finite], mu[is_finite])
    if upper:
        a_k = 1 - a_k
        a_ref = 1 - a_ref
    limit[is_finite] = a_k / a_ref
    limit[np.isclose(a_k, 0.)] = np.inf
    return limit


def calc_p_alpha_ratio(mu, k):
    is_upper = k > mu
    ratio = np.zeros_like(mu)
    ratio[is_upper] = __calc_p_alpha__(mu[is_upper],
                                       k[is_upper],
                                       upper=True)
    ratio[~is_upper] = __calc_p_alpha__(mu[~is_upper],
                                        k[~is_upper],
                                        upper=False)
    return ratio

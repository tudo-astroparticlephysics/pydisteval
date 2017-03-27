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


def map_aggarwal_ratio(y_values, y_0=1., upper=True):
    flattened_y = np.copy(y_values.reshape(np.prod(y_values.shape)))
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    if len(finite_y) == 0:
        return y_values, 0.
    finite_y[finite_y > y_0] = np.NaN
    finite_y = np.log10(finite_y)
    y_min = np.min(finite_y)
    y_min *= 1.1
    finite_y /= y_min
    transformed_values = np.copy(flattened_y)
    transformed_values[finite] = finite_y

    is_nan = np.isnan(flattened_y)
    is_pos_inf = np.isposinf(flattened_y)
    is_neg_inf = np.isneginf(flattened_y)
    if upper:
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = np.inf
        transformed_values[is_neg_inf] = -np.inf
    else:
        transformed_values[finite] *= -1.
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = np.inf
        transformed_values[is_neg_inf] = -np.inf
    transformed_values = transformed_values.reshape(y_values.shape)
    return transformed_values, y_min


def map_aggarwal_limits(y_values, y_0=1., upper=True):
    flattened_y = np.copy(y_values.reshape(np.prod(y_values.shape)))
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    if len(finite_y) == 0:
        return y_values, 0.
    finite_y[finite_y > y_0] = np.NaN
    finite_y = np.log10(finite_y)
    y_min = np.min(finite_y)
    y_min *= 1.1
    finite_y /= y_min
    transformed_values = np.copy(flattened_y)
    transformed_values[finite] = finite_y

    is_nan = np.isnan(flattened_y)
    is_pos_inf = np.isposinf(flattened_y)
    is_neg_inf = np.isneginf(flattened_y)
    if upper:
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = np.inf
        transformed_values[is_neg_inf] = 0.
    else:
        transformed_values[finite] *= -1.
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = -8000
        transformed_values[is_neg_inf] = -8000
    transformed_values = transformed_values.reshape(y_values.shape)
    return transformed_values, y_min


def rescale_ratio(values, y_min, y_min_wanted):
    values = np.copy(values)
    finite = np.isfinite(values)
    factor = y_min / y_min_wanted
    values[finite] *= factor
    finite_values = values[finite]
    finite_values[np.absolute(finite_values) > 1] = np.inf
    values[finite] = finite_values
    return values


def rescale_limit(values, y_min, y_min_wanted):
    values = np.copy(values)
    finite = np.isfinite(values)
    factor = y_min / y_min_wanted
    values[finite] *= factor
    return values


def calc_p_alpha_limits(mu, rel_std):
    abs_std = np.zeros_like(rel_std)
    limits = np.zeros_like(rel_std)
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 0]
        returned_vals = __calc_p_alpha__(mu, abs_std, upper=False)

        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        is_zero_k = abs_std == 0.
        only_zero_mu = np.logical_and(np.logical_and(~is_zero_k, is_zero_mu),
                                      ~is_nan)
        only_zero_k = np.logical_and(np.logical_and(is_zero_k, ~is_zero_mu),
                                     ~is_nan)
        both_zero = np.logical_and(is_zero_k, is_zero_mu)

        returned_vals[both_zero] = np.nan
        returned_vals[only_zero_mu] = -np.inf
        returned_vals[only_zero_k] = np.inf
        limits[:, i, 0] = returned_vals
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 1]
        returned_vals = __calc_p_alpha__(mu, abs_std, upper=True)
        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        is_zero_k = abs_std == 0.
        only_zero_mu = np.logical_and(np.logical_and(~is_zero_k, is_zero_mu),
                                      ~is_nan)
        only_zero_k = np.logical_and(np.logical_and(is_zero_k, ~is_zero_mu),
                                     ~is_nan)
        both_zero = np.logical_and(is_zero_k, is_zero_mu)

        returned_vals[both_zero] = np.nan
        returned_vals[only_zero_mu] = -np.inf
        returned_vals[only_zero_k] = np.inf
        limits[:, i, 1] = returned_vals
    return limits


def __calc_p_alpha__(mu, k, upper=True):
    assert mu.shape == k.shape, 'Shape of \'mu\' and \'k\' have to be the same'
    limit = np.copy(k)
    is_zero_mu = mu == 0.
    is_zero_k = k == 0.

    is_nan = np.logical_or(np.isnan(k), np.isnan(mu))
    is_finite = np.logical_and(~is_zero_k, ~is_zero_mu)

    a_ref = sc_dist.poisson.cdf(mu[is_finite], mu[is_finite])
    a_k = sc_dist.poisson.cdf(k[is_finite], mu[is_finite])
    if upper:
        ratio = (1 - a_k) / (1 - a_ref)
        ratio[1 - a_k == 0.] = np.inf
    else:
        ratio = a_k / a_ref
        ratio[a_k == 0.] = np.inf
    limit[is_finite] = ratio
    limit[is_nan] = np.nan
    return limit


def calc_p_alpha_ratio(mu, k):
    is_upper = k > mu
    ratio = np.zeros_like(mu)
    for upper in [False, True]:
        if upper:
            mask = is_upper
        else:
            mask = ~is_upper
        returned_vals = __calc_p_alpha__(mu[mask],
                                         k[mask],
                                         upper=upper)
        is_nan = np.logical_or(np.isnan(k[mask]), np.isnan(mu[mask]))
        is_zero_mu = mu[mask] == 0.
        is_zero_k = k[mask] == 0.
        only_zero_mu = np.logical_and(np.logical_and(~is_zero_k, is_zero_mu),
                                      ~is_nan)
        only_zero_k = np.logical_and(np.logical_and(is_zero_k, ~is_zero_mu),
                                     ~is_nan)
        both_zero = np.logical_and(is_zero_k, is_zero_mu)
        if upper:
            returned_vals[both_zero] = np.nan
            returned_vals[only_zero_mu] = -np.inf
            returned_vals[only_zero_k] = np.inf
        else:
            returned_vals[both_zero] = np.nan
            returned_vals[only_zero_mu] = -np.inf
            returned_vals[only_zero_k] = np.inf
        ratio[mask] = returned_vals
    return ratio

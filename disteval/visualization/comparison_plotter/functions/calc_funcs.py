import numpy as np
import scipy.stats.distributions as sc_dist
from itertools import compress


def aggarwal_limits(mu, alpha=0.68268949):
    """Get Poissonian limits for specified contour levels

    Parameters
    ----------
    pdfs : array_like
        The expected number of events (Poisson mean) in each observable bin.
        Shape: [n_bins]
    alpha : float or list of float, optional
        The list of alpha values, which define the contour levels which will
        be computed.

    Returns
    -------
    array_like
        The lower limits (minus -0.5) for each of the observable bins and
        chosen contour value alpha.
        Shape: [n_bins, n_alpha]
    array_like
        The upper limits (plus +0.5) for each of the observable bins and
        chosen contour value alpha.
        Shape: [n_bins, n_alpha]
    """
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
    lower[lower != 0] -= 0.5
    upper += 0.5
    return lower.reshape(mu_large.shape), upper.reshape(mu_large.shape)


def aggarwal_limits_pdf(pdfs, ks, alpha=0.68268949):
    """Get limits for specified contour levels

    In contrast to `aggarwal_limits` this function computes the limits based
    on the evaluated and normalized likelihood as opposed to the theoretical
    limits from the Poisson disribution.

    Parameters
    ----------
    pdfs : list of list of float
        The pdf values for each feature bin and for each value k.
        The value k is the observed number of events in the Poisson Likelihood.
        The number of evaluted k values is different for each observable bin,
        and it is chosen such that a certain coverage is obtained.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    ks : list of list of int
        The corresponding k value for each of the evaluated pdf values `pdfs`.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    alpha : float or list of float, optional
        The list of alpha values, which define the contour levels which will
        be computed.

    Returns
    -------
    array_like
        The lower limits (minus -0.5) for each of the observable bins and
        chosen contour value alpha.
        Shape: [n_bins, n_alpha]
    array_like
        The upper limits (plus +0.5) for each of the observable bins and
        chosen contour value alpha.
        Shape: [n_bins, n_alpha]
    """
    if isinstance(alpha, float):
        alpha = [alpha]
    lower = np.zeros((len(pdfs), len(alpha)))
    upper = np.zeros((len(pdfs), len(alpha)))

    for i, pdf in enumerate(pdfs):
        if len(ks[i]) == 0:
            continue
        cdf = np.cumsum(pdf)
        if cdf[-1] < 0.999:
            print('Cdf only goes up to {}'.format(cdf[-1]))
            lower[i, :] = np.nan
            upper[i, :] = np.nan
            continue
        for j, alpha_j in enumerate(alpha):
            q1 = (1.-alpha_j) / 2.
            q2 = (1.+alpha_j) / 2.
            lower_idx = np.searchsorted(cdf, q1)
            upper_idx = np.searchsorted(cdf, q2)

            lower[i, j] = ks[i][lower_idx]
            upper[i, j] = ks[i][upper_idx]

    lower[lower != 0] -= 0.5
    upper += 0.5
    return lower, upper


def evaluate_normalized_likelihood(llh_func, coverage,
                                   first_guess, **llh_kwargs):
    """Compute normalized likelihood

    This function evaluates the likelihood function `llh_func` iteratively over
    possible values of k (observed number of events in Poissonian) until
    the specified coverage is reached.
    This can then be used to normalize the likelihood and to define the PDF
    in observed values k and to compute the limits in k.

    Parameters
    ----------
    llh_func : callable
        The likelihood function
    coverage : float
        The minimum coverage value to obtain. Max value is 1. The closer to
        1, the more accurate, but also more time consuming.
    first_guess : float
        A first guess of the valid range of k values. Typically, this can
        be set to the expected number of values in the observable bin.
    **llh_kwargs
        Keyword arguments that are passed on to the likelihood function.

    Returns
    -------
    array_like
        The (sorted) k values at which the likelhood was evaluted.
    array_like
        The corresponding likelihood values to each of the (sorted) k values.
        These are normalized, i.e. their sum should approach 1, but be at
        least as high as the specified `coverage`.
    """
    mu = int(first_guess)
    prob = np.exp(llh_func(mu, **llh_kwargs))
    unsorted_pdf = [prob]
    ks = [mu]
    max_k = mu
    min_k = mu

    reached_bottom = False

    while prob < coverage:
        if not reached_bottom:
            if min_k == 0:
                reached_bottom = True
            else:
                min_k -= 1
                ks.append(min_k)
                new_val = np.exp(llh_func(min_k, **llh_kwargs))
                unsorted_pdf.append(
                    new_val)
                prob += new_val

        max_k += 1
        ks.append(max_k)
        new_val = np.exp(llh_func(max_k, **llh_kwargs))
        unsorted_pdf.append(new_val)
        prob += new_val

    ks = np.array(ks)
    unsorted_pdf = np.array(unsorted_pdf)
    sort_idx = np.argsort(ks)
    sorted_ks = ks[sort_idx]
    sorted_pdf = unsorted_pdf[sort_idx]
    return sorted_ks, sorted_pdf


def map_aggarwal_ratio(y_values, y_0=1., upper=True):
    """Map p-values to relative y-values wrt minimium p-value.

    The provided p-values `y_values` are mapped to relative y-values.
    These transformed y-values are relative to the minimum p-value (in log10).
    Depending on whether or not `upper` is True, these relative values will
    either be positive or negative. In other words, the p-values are mapped
    to y-values in the range of [0, 1] for upper == True and [-1, 0] for
    upper == False.

    Parameters
    ----------
    y_values : array_like
        The p-values for each observable bin.
        Shape: [n_bins]
    y_0 : float, optional
        The highest possible p-value. Anything above this is set to NaN, i.e.
        it will not be plotted later.
    upper : bool, optional
        If True, the ratios are above the expectation values, i.e. the
        transformed values will be in the range of [0, 1].
        If False, the ratios are below the expectation values in each bin
        and the transformed values will be in the range of [-1, 0].

    Returns
    -------
    array_like
        The transformed y-values for each of the p-values `y_values`.
        Shape: [n_bins]
    """
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
    got_divided_by_zero = flattened_y == 1.
    if upper:
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = np.inf
        transformed_values[is_neg_inf] = -np.inf
    else:
        transformed_values[finite] *= -1.
        transformed_values[is_nan] = np.nan
        transformed_values[is_pos_inf] = np.inf
        transformed_values[is_neg_inf] = -np.inf
    transformed_values[got_divided_by_zero] = 0
    transformed_values = transformed_values.reshape(y_values.shape)
    return transformed_values, y_min


def map_aggarwal_limits(y_values, y_0=1., upper=True):
    """Map p-values to relative y-values wrt minimium p-value.

    The provided p-values `y_values` are mapped to relative y-values.
    These transformed y-values are relative to the minimum p-value (in log10).
    Depending on whether or not `upper` is True, these relative values will
    either be positive or negative. In other words, the p-values are mapped
    to y-values in the range of [0, 1] for upper == True and [-1, 0] for
    upper == False.

    This function is similar to `map_aggarwal_ratio`, but the handling
    of positive and negative infinities are different. These are set to finite
    values, such that appropriate limit contours may be drawn.

    Parameters
    ----------
    y_values : array_like
        The p-values for each observable bin.
        Shape: [n_bins]
    y_0 : float, optional
        The highest possible p-value. Anything above this is set to NaN, i.e.
        it will not be plotted later.
    upper : bool, optional
        If True, the limits are upper limits, i.e. the
        transformed values will be in the range of [0, 1].
        If False, the limits are lower limits and the transformed values will
        be in the range of [-1, 0].

    Returns
    -------
    array_like
        The transformed y-values for each of the p-values `y_values`.
        Shape: [n_bins]
    """
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
    """Rescale relative y-values

    Rescales relative y-values `values` to `y_min_wanted`. It is assumed
    that the provied values are relative to the minimum p-value as specified
    in the provided `y_min`.

    Similar to `rescale_limit`, but does additional handling of points that
    are outside of the plot region (these are set to inf, such that they will
    not be plotted).

    Parameters
    ----------
    values : array_like
        The relative y-values that should be rescaled.
        Shape: [n_bins]
    y_min : float
        The minimum p-value. This is the anchor point to which the original
        p-values were scaled to, i.e. `values` are relative to this minimum
        p-value.
        Shape: []
    y_min_wanted : flaot
        The desired new minimum p-value. This is the new anchor point to which
        the original p-values will be re-scaled to.
        Shape: []

    Returns
    -------
    array_like
        The rescaled y-values now relative to `y_min_wanted`.
        Shape: [n_bins]
    """
    values = np.copy(values)
    finite = np.isfinite(values)
    factor = y_min / y_min_wanted
    values[finite] *= factor
    finite_values = values[finite]
    finite_values[np.absolute(finite_values) > 1] = np.inf
    values[finite] = finite_values
    return values


def rescale_limit(values, y_min, y_min_wanted):
    """Rescale relative y-values

    Rescales relative y-values `values` to `y_min_wanted`. It is assumed
    that the provied values are relative to the minimum p-value as specified
    in the provided `y_min`.

    Parameters
    ----------
    values : array_like
        The relative y-values that should be rescaled.
        Shape: [n_bins]
    y_min : float
        The minimum p-value. This is the anchor point to which the original
        p-values were scaled to, i.e. `values` are relative to this minimum
        p-value.
        Shape: []
    y_min_wanted : flaot
        The desired new minimum p-value. This is the new anchor point to which
        the original p-values will be re-scaled to.
        Shape: []

    Returns
    -------
    array_like
        The rescaled y-values now relative to `y_min_wanted`.
        Shape: [n_bins]
    """
    values = np.copy(values)
    finite = np.isfinite(values)
    factor = y_min / y_min_wanted
    values[finite] *= factor
    return values


def calc_p_alpha_limits(mu, rel_std):
    """Get the CDF ratio at the limits `rel_std` in each observable bin.

    Parameters
    ----------
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    rel_std : array_like
        The relative limits wrt the expected number (Poisson mean) of events
        in each bin, i.e. limits / mu. The last dimension corresponds to lower
        and upper relative limits, respectively.
        Shape: [n_bins, n_alpha, 2]

    array_like
        The ratio of the PDF tails:
            P(x <= limit_i) / P(x <= mu_i) if limit_i <= mu_i
            P(x > limit_i) / P(x > mu_i) if limit_i > mu_i
        for each observable bin i.
        The CDF P(x <= y) is calculated based on the expected number of events
        in each observable bin and under the assumption of a Poisson
        distribution.
        This ratio reaches 1., if the measured values `k` agree well with the
        expected values `mu`. The smaller this ratio is, the higher the
        discrepancy.
        Shape: [n_bins, n_alpha, 2]
    """
    abs_std = np.zeros_like(rel_std)
    limits = np.zeros_like(rel_std)
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 0]
        returned_vals = __calc_p_alpha__(mu, abs_std, upper=False)

        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)

        returned_vals[only_zero_mu] = -np.inf
        limits[:, i, 0] = returned_vals
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 1]
        returned_vals = __calc_p_alpha__(mu, abs_std, upper=True)
        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        is_zero_k = abs_std == 0.
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)

        returned_vals[only_zero_mu] = -np.inf
        limits[:, i, 1] = returned_vals
    return limits


def calc_p_alpha_limits_pdf(pdfs, ks, mu, rel_std):
    """Get the CDF ratio at the limits `rel_std` in each observable bin.

    Similar to `calc_p_alpha_limits`, but the CDF calculation is based on the
    normalized likelihood values `pdfs` and corresponding k values `ks`.

    Parameters
    ----------
    pdfs : list of list of float
        The pdf values for each feature bin and for each value k.
        The value k is the observed number of events in the Poisson Likelihood.
        The number of evaluted k values is different for each observable bin,
        and it is chosen such that a certain coverage is obtained.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    ks : list of list of int
        The corresponding k value for each of the evaluated pdf values `pdfs`.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    rel_std : array_like
        The relative limits wrt the expected number (Poisson mean) of events
        in each bin, i.e. limits / mu. The last dimension corresponds to lower
        and upper relative limits, respectively.
        Shape: [n_bins, n_alpha, 2]

    Returns
    -------
    array_like
        The ratio of the PDF tails:
            P(x <= limit_i) / P(x <= mu_i) if limit_i <= mu_i
            P(x > limit_i) / P(x > mu_i) if limit_i > mu_i
        for each observable bin i.
        The CDF P(x <= y) is calculated based on the normalized likelihood
        values `pdfs` and corresponding k values `ks`.
        This ratio reaches 1., if the measured values `k` agree well with the
        expected values `mu`. The smaller this ratio is, the higher the
        discrepancy.
        Shape: [n_bins, n_alpha, 2]
    """
    abs_std = np.zeros_like(rel_std)
    limits = np.zeros_like(rel_std)
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 0]
        returned_vals = __calc_p_alpha_pdf__(pdfs, ks,
                                             mu, abs_std,
                                             upper=False)

        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)

        returned_vals[only_zero_mu] = -np.inf
        limits[:, i, 0] = returned_vals
    for i in range(rel_std.shape[1]):
        abs_std = mu * rel_std[:, i, 1]
        returned_vals = __calc_p_alpha_pdf__(pdfs, ks,
                                             mu, abs_std,
                                             upper=True)
        is_nan = np.logical_or(np.isnan(abs_std), np.isnan(mu))
        is_zero_mu = mu == 0.
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)

        returned_vals[only_zero_mu] = -np.inf
        limits[:, i, 1] = returned_vals
    return limits


def __calc_p_alpha__(mu, k, upper=True):
    """Get the CDF ratio at a given number of observed events k in each bin.

    Parameters
    ----------
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    k : array_like
        The measured number (Poisson k) of events in each observable bin.
        The CDF ratio is evaluated at these k values.
        Shape: [n_bins]
    upper : bool, optional
        If true, the upper PDF tail will be considered, i.e. the ratio
        P(x > k_i) / P(x > mu_i) will be computed.
        If false, P(x <= k_i) / P(x <= mu_i) is computed.

    Returns
    -------
    array_like
        The ratio P(x <= k_i) / P(x <= mu_i) for each observable bin i.
        The CDF P(x <= y) is calculated based on the expected number of events
        in each observable bin and under the assumption of a Poisson
        distribution. If upper is True, then '<=' switches to '>'.
        Shape: [n_bins]
    """
    assert mu.shape == k.shape, 'Shape of \'mu\' and \'k\' have to be the same'
    limit = np.copy(k)

    is_nan = np.logical_or(np.isnan(k), np.isnan(mu))
    is_finite = mu != 0.

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


def __calc_p_alpha_pdf__(pdfs, ks, mu, k, upper=True):
    """Get the CDF ratio at a given number of observed events k in each bin.

    Similar to `__calc_p_alpha__`, but CDF is calculated based on the
    computed normalized likelihood values `pdfs` and the corresponding
    k values `ks`.

    Parameters
    ----------
    pdfs : list of list of float
        The pdf values for each feature bin and for each value k.
        The value k is the observed number of events in the Poisson Likelihood.
        The number of evaluted k values is different for each observable bin,
        and it is chosen such that a certain coverage is obtained.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    ks : list of list of int
        The corresponding k value for each of the evaluated pdf values `pdfs`.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    k : array_like
        The measured number (Poisson k) of events in each observable bin.
        The CDF ratio is evaluated at these k values.
        Shape: [n_bins]
    upper : bool, optional
        If true, the upper PDF tail will be considered, i.e. the ratio
        P(x > k_i) / P(x > mu_i) will be computed.
        If false, P(x <= k_i) / P(x <= mu_i) is computed.

    Returns
    -------
    array_like
        The ratio P(x <= k_i) / P(x <= mu_i) for each observable bin i.
        The CDF P(x <= y) is calculated based on the normalized likelihood
        values `pdfs` and corresponding k values `ks`.
        If upper is True, then '<=' switches to '>'.
        Shape: [n_bins]
    """
    assert mu.shape == k.shape, 'Shape of \'mu\' and \'k\' have to be the same'
    limit = np.copy(k)

    is_nan = np.logical_or(np.isnan(k), np.isnan(mu))
    is_finite = mu != 0.

    for i, (pdf, ksi) in enumerate(zip(pdfs, ks)):
        cdf = np.cumsum(pdf)
        if is_finite[i]:
            mu_idx = np.where(ksi == int(mu[i]))[0]
            if len(mu_idx) == 0:
                a_ref = np.nan
            else:
                a_ref = cdf[mu_idx]
            k_idx = np.where(ksi == int(k[i]))[0]
            if len(k_idx) == 0:
                if upper:
                    a_k = 1
                else:
                    a_k = 0
            else:
                a_k = cdf[k_idx]
            if upper:
                if 1 - a_k == 0.:
                    limit[i] = np.inf
                else:
                    ratio = (1 - a_k) / (1 - a_ref)
                    limit[i] = ratio
            else:
                if a_k == 0:
                    limit[i] = np.inf
                else:
                    ratio = a_k / a_ref
                    limit[i] = ratio

    limit[is_nan] = np.nan
    return limit


def calc_p_alpha_ratio(mu, k):
    """Get the CDF ratio at the measured `k` values in each observable bin.

    Parameters
    ----------
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    k : array_like
        The measured number (Poisson k) of events in each observable bin.
        Shape: [n_bins]

    array_like
        The ratio of the PDF tails:
            P(x <= k_i) / P(x <= mu_i) if k_i <= mu_i
            P(x > k_i) / P(x > mu_i) if k_i > mu_i
        for each observable bin i.
        The CDF P(x <= y) is calculated based on the expected number of events
        in each observable bin and under the assumption of a Poisson
        distribution.
        This ratio reaches 1., if the measured values `k` agree well with the
        expected values `mu`. The smaller this ratio is, the higher the
        discrepancy.
        Shape: [n_bins]
    """
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
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)
        returned_vals[only_zero_mu] = -np.inf
        ratio[mask] = returned_vals
    return ratio


def calc_p_alpha_ratio_pdf(pdfs, ks, mu, k):
    """Get the CDF ratio at the measured `k` values in each observable bin.

    Similar to `calc_p_alpha_ratio`, but the CDF calculation is based on the
    normalized likelihood values `pdfs` and corresponding k values `ks`.

    Parameters
    ----------
    pdfs : list of list of float
        The pdf values for each feature bin and for each value k.
        The value k is the observed number of events in the Poisson Likelihood.
        The number of evaluted k values is different for each observable bin,
        and it is chosen such that a certain coverage is obtained.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    ks : list of list of int
        The corresponding k value for each of the evaluated pdf values `pdfs`.
        Shape: [n_bins, n_k_values] (note that n_k_values is not constant!)
    mu : array_like
        The expected number (Poisson mean) of events in each observable bin.
        Shape: [n_bins]
    k : array_like
        The measured number (Poisson k) of events in each observable bin.
        Shape: [n_bins]

    Returns
    -------
    array_like
        The ratio of the PDF tails:
            P(x <= k_i) / P(x <= mu_i) if k_i <= mu_i
            P(x > k_i) / P(x > mu_i) if k_i > mu_i
        for each observable bin i.
        The CDF P(x <= y) is calculated based on the normalized likelihood
        values `pdfs` and corresponding k values `ks`.
        This ratio reaches 1., if the measured values `k` agree well with the
        expected values `mu`. The smaller this ratio is, the higher the
        discrepancy.
        Shape: [n_bins]
    """
    is_upper = k > mu
    ratio = np.zeros_like(mu)
    for upper in [False, True]:
        if upper:
            mask = is_upper
        else:
            mask = ~is_upper
        returned_vals = __calc_p_alpha_pdf__(compress(pdfs, mask),
                                             compress(ks, mask),
                                             mu[mask],
                                             k[mask],
                                             upper=upper)
        is_nan = np.logical_or(np.isnan(k[mask]), np.isnan(mu[mask]))
        is_zero_mu = mu[mask] == 0.
        only_zero_mu = np.logical_and(is_zero_mu, ~is_nan)
        returned_vals[only_zero_mu] = -np.inf
        ratio[mask] = returned_vals
    return ratio

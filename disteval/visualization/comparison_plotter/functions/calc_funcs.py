import numpy as np
import scipy.stats.distributions as sc_dist


def aggarwall_limits(mu, alpha=0.68268949):
    if isinstance(alpha, float):
        alpha = [alpha]
    lim = np.zeros((len(mu), len(alpha), 2))
    mu_large = np.zeros((len(mu), len(alpha)))
    alpha_large = np.zeros_like(mu_large)
    for i, a_i in enumerate(alpha):
        alpha_large[:, i] = a_i
        mu_large[:, i] = mu

    mu_large_flat = mu_large.reshape(np.prod(mu_large.shape))
    alpha_large_flat = alpha_large.reshape(mu_large_flat.shape)
    lower, upper = sc_dist.poisson.interval(alpha_large_flat, mu_large_flat)
    return lower.reshape(mu_large.shape), upper.reshape(mu_large.shape)

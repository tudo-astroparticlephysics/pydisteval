import numpy as np
import scipy


def poisson_equal_weights(k, k_mc,
                          avgweights,
                          prior_factor=0.0):
    return (scipy.special.gammaln((k + k_mc + prior_factor)) -
            scipy.special.gammaln(k + 1.0) -
            scipy.special.gammaln(k_mc + prior_factor) +
            (k_mc + prior_factor) * np.log(1.0 / avgweights) -
            (k_mc + k + prior_factor) * np.log(1.0 + 1.0 / avgweights)).sum()


def poisson_general_weights(k, weights, prior=0.0):
    # treat each weight independently without any more checks
    # (see github readme exaxmple)
    if len(weights) == 0:
        return np.nan

    weight_prefactors = (-(1.0+prior/float(len(weights))) *
                         np.log(weights)).sum()

    new_zs = 1. + 1. / weights

    new_zs_log = np.log(new_zs)
    new_bs = np.ones(len(new_zs), dtype=float)
    new_bs += prior * new_bs / float(len(weights))

    res = (-new_bs * new_zs_log).sum()

    cs = [res]
    if(k > 0):
        lambdas = []

        new_bs_log = np.log(new_bs)
        running_lambda_vec = new_bs_log

        for cur_ind in range(k):
            running_lambda_vec -= new_zs_log
            lambdas.append(scipy.misc.logsumexp(running_lambda_vec).sum())

            new_cs = scipy.misc.logsumexp(
                np.array(lambdas[::-1]) + np.array(cs)) - np.log(cur_ind+1)
            cs.append(new_cs)

    return weight_prefactors+cs[-1]


def poisson_general_weights_chirkin_13(data, all_weights, weight_indices):
    """
    Returns the positive log-likelihood value between data and simulation
    taking into account the finite statistics.
    """
    def func(x, w, d):
        """
        Reweighting function: 1/sum(w/(1+xw))
        The function should be equal to (1-x)/N_exp
        for reweighting variable x. Note, that w is an array
        since it is the (i,j) entry of w_hist.
        """
        return 1./np.sum(w/(1. + x*w)) - (1. - x)/d

    if data.ndim == 1:
        # array of reweighting factors
        lagrange = np.array(
            [(scipy.optimize.brentq(
                func, -0.999999/max(all_weights[w]),
                1.,
                args=(all_weights[w], d),
                full_output=False)
             if d else 1.) if (len(w) > 0) else 0.
             for (d, w) in zip(data, weight_indices)])
        # llh with new weights
        llh = np.array(
            [np.sum(np.log(1. + lagrange[i]*all_weights[w]))
             if(len(w) > 0) else 0 for (i, w) in enumerate(weight_indices)]) \
            + data * np.log(1.-(lagrange-(lagrange == 1.)))
    else:
        raise NotImplementedError("`data` has more than 1 dimensions.")

    return -llh.sum()


def poisson_llh(k, w_sum):
    return -scipy.stats.poisson.logpmf(k, mu=w_sum)


def gamma_prior_poisson_llh(k, alpha, beta):
    terms = []
    terms.append(alpha * np.log(beta))
    terms.append(scipy.special.gammaln(k+alpha))
    terms.append(-scipy.special.gammaln(k+1))
    terms.append(-(k+alpha) * np.log1p(beta))
    terms.append(-scipy.special.gammaln(alpha))
    return np.sum(terms)


def SAY_likelihood(k, mu, w2_sum):
    w_sum = mu
    if w_sum <= 0 or w2_sum <= 0:
        if k == 0:
            return 0
        else:
            return -np.inf

    if w2_sum == 0:
        return poisson_llh(k, w_sum)

    mu = w_sum
    mu2 = mu**2
    sigma2 = w2_sum

    beta = (mu + np.sqrt(mu2 + sigma2 * 4)) / (2.*sigma2)
    alpha = (mu * np.sqrt(mu2 + sigma2 * 4) / sigma2 + mu2 / sigma2 + 2) / 2.

    llh = gamma_prior_poisson_llh(k, alpha, beta)

    return llh


def SAY_likelihood_bayes(k, w_sum, w2_sum):
    if w_sum <= 0 or w2_sum <= 0:
        if k == 0:
            return 0
        else:
            return -np.inf

    if w2_sum == 0:
        print('wat')
        return poisson_llh(k, w_sum)

    alpha = w_sum ** 2 / w2_sum
    beta = w_sum / w2_sum

    llh = gamma_prior_poisson_llh(k, alpha, beta)

    return llh

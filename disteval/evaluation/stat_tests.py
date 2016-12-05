# -*- coding:utf-8 -*-
"""
Collection of methods to evaluate the results of disteval functions
"""
import numpy as np


def kstest_2sample(x, cdf_a, cdf_b, n_a, n_b, alpha=0.05):
    """Function evaluating the Kolmogorov Smirrnoff Test. Variable
    naming orianted torwards the

    Parameters
    ----------
    x: numpy.array, shape=(N,)
        Array of all x value position corresponding to the CDF values
        for both samples.

    cdf_a: numpy.array, shape=(N,)
        CDF values for sample a.

    cdf_b: numpy.array, shape=(N,)
        CDF values for sample b.

    n_a: int
        Number of observations in sample a.

    n_b: int
        Number of observations in sample b.


    alpha : float, optional (default=0.05)
        Significance for the Kolmogorov Smirnov test.

    Returns
    -------
    passed: bool
        True if test is accepted. False if the test is rejected. A
        rejection has the error rate alpha.

    idx_max: int
        Index of the largest distance. x[idx_max] is the x position for
        the largest distance.

    d_max: float
        Largest distance between both sample cdfs.
    """
    d = np.absolute(cdf_a - cdf_b)
    idx_max = np.argmax(d)
    d_max = d[idx_max]

    K_alpha = np.sqrt(np.log(2. / np.sqrt(alpha)) / 2)
    factor = np.sqrt(n_a * n_b / (n_a + n_b))
    passed = factor * d_max <= K_alpha

    return passed, idx_max, d_max

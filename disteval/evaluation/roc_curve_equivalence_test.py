# -*- coding:utf-8 -*-
"""
Collection of methods to evaluate the results of disteval functions
"""
from __future__ import absolute_import, print_function, division
import numpy as np

from sklearn.metrics import roc_curve

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


def roc_curve_equivalence_ks_test(y_pred_a,
                                  y_pred_b,
                                  y_true,
                                  y_true_b=None,
                                  alpha=0.05,
                                  scale=False):
    """Function evaluating the equivalence between the ROC curves of
    two classifier. The method is described by Andrew P. Bradley in
    "ROC curve equivalence using the Kolmogorov-Smirnov test"
    DOI: 10.1016/j.patrec.2012.12.021

    Parameters
    ----------
    y_pred_a: numpy.array, shape=(n_samples_a)
        Predictions of classifier a. The predictions are expected to be
        between [0, 1].

    y_pred_b: numpy.array, shape=(n_samples_b)
        Predictions of classifier b. he predictions are expected to be
        between [0, 1]. If y_true_b is not provided, the
        sample must be of the same length as sample a.

    y_true : numpy.array, shape=(n_samples_a)
        True labels for sample_a. If y_true_b is not provided, it is
        also used as the true labels for sample b

    y_true_b : None numpy.array, shape=(n_samples_b), optional
        True labels for sample_b. If None y_true is used as labels for
        sample b.

    alpha : float, optional (default=0.05)
        Significance for the Kolmogorov Smirnov test.

    scale : boolean, optional (default=False)
        Wether the predictions should be to the interval [0,1].

    Returns
    -------
    passed: bool
        True if test is accepted. False if the test is rejected. A
        rejection has the error rate alpha.

    op_point_a: numpy.array, shape=(2,2)
        [False positive rate, True positive rate] Rate at the operation
        points of both KS test for sample a.

    op_point_b: numpy.array, shape=(2,2)
        [False positive rate, True positive rate] Rate at the operation
        points of both KS test for sample b.

    fpr_b: numpy.array
        False positive rate for sample b at the thresholds.

    tpr_b: numpy.array
        True positive rate for sample b at the thresholds.

    threshold: numpy.array
        Thresholds to the false/true positive rates.
    """

    bincount_y = np.bincount(y_true)
    num_positive_a = bincount_y[1]
    num_negative_a = bincount_y[0]
    if y_true_b is not None:
        bincount_y = np.bincount(y_true_b)
        num_positive_b = bincount_y[1]
        num_negative_b = bincount_y[0]
    else:
        y_true_b = y_true
        num_positive_b = num_positive_a
        num_negative_b = num_negative_a
    if scale:
        min_pred_a = np.min(y_pred_a)
        max_pred_a = np.max(y_pred_a)
        y_pred_a = (y_pred_a - min_pred_a) / (max_pred_a - min_pred_a)

        min_pred_b = np.min(y_pred_b)
        max_pred_b = np.max(y_pred_b)
        y_pred_b = (y_pred_b - min_pred_b) / (max_pred_b - min_pred_b)


    fpr_a, tpr_a, thresholds_a = roc_curve(y_true,
                                           y_pred_a,
                                           drop_intermediate=True)
    fpr_b, tpr_b, thresholds_b = roc_curve(y_true_b,
                                           y_pred_b,
                                           drop_intermediate=True)

    thresholds = np.sort(np.unique(np.hstack((thresholds_a, thresholds_b))))

    order_a = np.argsort(thresholds_a)
    thresholds_a = thresholds_a[order_a]
    fpr_a = fpr_a[order_a]
    tpr_a = tpr_a[order_a]

    order_b = np.argsort(thresholds_b)
    thresholds_b = thresholds_b[order_b]
    fpr_b = fpr_b[order_b]
    tpr_b = tpr_b[order_b]

    fpr_a_full = np.ones_like(thresholds)
    tpr_a_full = np.ones_like(thresholds)
    fpr_b_full = np.ones_like(thresholds)
    tpr_b_full = np.ones_like(thresholds)
    pointer_a = -1
    pointer_b = -1

    for i, t_i in enumerate(thresholds):
        if pointer_a + 1 < len(thresholds_a):
            if t_i == thresholds_a[pointer_a + 1]:
                pointer_a += 1
            fpr_a_full[i] = fpr_a[pointer_a]
            tpr_a_full[i] = tpr_a[pointer_a]
            if pointer_a == -1:
                fpr_a_full[i] = 1.
                tpr_a_full[i] = 1.
        else:
            fpr_a_full[i] = 0.
            tpr_a_full[i] = 0.

        if pointer_b + 1 < len(thresholds_b):
            if t_i == thresholds_b[pointer_b + 1]:
                pointer_b += 1
            fpr_b_full[i] = fpr_b[pointer_b]
            tpr_b_full[i] = tpr_b[pointer_b]
            if pointer_b == -1:
                fpr_b_full[i] = 1.
                tpr_b_full[i] = 1.
        else:
            fpr_b_full[i] = 0.
            tpr_b_full[i] = 0.

    passed_neg, idx_max_neg, dist_max_neg = kstest_2sample(
        x=thresholds,
        cdf_a=fpr_a_full,
        cdf_b=fpr_b_full,
        n_a=num_negative_a,
        n_b=num_negative_b,
        alpha=alpha)

    passed_pos, idx_max_pos, dist_max_pos = kstest_2sample(
        x=thresholds,
        cdf_a=tpr_a_full,
        cdf_b=tpr_b_full,
        n_a=num_positive_a,
        n_b=num_positive_b,
        alpha=alpha)

    op_point_n = np.array([[fpr_a_full[idx_max_neg], fpr_b_full[idx_max_neg]],
                           [tpr_a_full[idx_max_neg], tpr_b_full[idx_max_neg]]])
    op_point_p = np.array([[fpr_a_full[idx_max_pos], fpr_b_full[idx_max_pos]],
                           [tpr_a_full[idx_max_pos], tpr_b_full[idx_max_pos]]])

    passed = np.logical_and(passed_pos, passed_neg)

    return passed, op_point_n, op_point_p, \
        fpr_a_full, tpr_a_full, fpr_b_full, tpr_b_full, thresholds

# -*- coding:utf-8 -*-
"""
Collection of methods to evaluate the results of disteval functions
"""
import numpy as np

from scipy.stats import norm
from sklearn.metrics import roc_curve

from ..scripts.classifier_characteristics import ClassifierCharacteristics
from .stat_tests import kstest_2sample

def feature_importance_mad(clf, alpha=0.05):
    """This function fetches the feature importance values and runs a
    criteria using the median absolute deviation. If a feature
    importance difference to the median importance is greater than
    a certain threshold and the feature is more important than the
    median feature, the feature is removed. The threshold is:
    1.4826 * cdf_norm**-1(1 - alpha/2) * MAD
    The distribution of the feature importance can be expected, to have
    a relativ flat distribution up from 0 upto a normal distributed
    peak. The flat part is for constant or close to constant features.
    The rest of the features can be expected to be choosen in a random
    fashion. Therefore they build a normal distributed peak
    around ~(1. / (n_features - n_constant_features)). To have a robust
    measure for outliers the meadian absolute diviation (MAD) is used.
    The definition of the MAD is:
    median(|X_i - median(X)|)
    For a mormal distribution the 1 sigma region is included in the
    region between 1.4826 * MAD - median(X) and 1.4826 * MAD + median(X).
    With the parameter alpha the used threshold is tuned in a way, for
    a pure normal distribution alpha / 2  (only features above the
    median are removed) features would be removed.

    Parameters
    ----------
    clf: object or list
        Trained classifier or list of trained classifier.

    alpha : float, optional (default=0.05)
        Parameter tuning the threshold. See function describtion.

    Returns
    -------
    kept: numpy.boolarray, shape=(n_features)
        Whether the feature passes the MAD criteria.

    feature_importance: numpy.array, shape=(n_features)
        Array of the importance values for the features. If a list of
        classifier is passed, it is the mean over all classifier.

    feature_importance_std: None or numpy.array, shape=(n_features)
        If a list of classifier is passed the standard deviation is of
        the feature importance values is returned. Otherwise None is
        returned
    """
    desired_characteristics = ClassifierCharacteristics()
    desired_characteristics.opts['has:feature_importances_'] = True

    if isinstance(clf, list):
        feature_importances = []
        for i, clf_i in enumerate(clf):
            clf_characteristics = ClassifierCharacteristics(clf_i)
            assert clf_characteristics.fulfilling(desired_characteristics), \
                'Classifier sanity check failed!'
            feature_importances.append(clf_i.feature_importances_)
        feature_importances = np.array(feature_importances)
        feature_importance = np.mean(feature_importances, axis=0)
        feature_importance_std = np.std(feature_importances, axis=0, ddof=1)
    else:
        clf_characteristics = ClassifierCharacteristics(clf)
        assert clf_characteristics.fulfilling(desired_characteristics), \
            'Classifier sanity check failed!'
        feature_importance = clf.feature_importances_
        feature_importance_std = np.NaN

    threshold = norm.ppf(1 - alpha/2) * 1.4826  # see docstring
    median_importance = np.median(feature_importance)
    MAD = np.median(np.absolute(feature_importance - median_importance))
    diff = feature_importance - median_importance
    kept = np.logical_or(np.absolute(diff) < threshold * MAD,
                         feature_importance <= median_importance)
    return kept, feature_importance, feature_importance_std


def feature_importance_mad_majority(clfs, ratio=0.9, alpha=0.10):
    """In this function a list of classifier must be provided. To decide
    if a feature is removed, for each classifier the function
    feature_importance_mad with the provided alpha is evaluated. And if
    a feature is removed in atleast ratio-percent of the classifiers
    the feature is removed. The motivation behind the majority vote is,
    that if a feature is just above the threshold in a single test
    because of statistical fluctuation is should be below the threshold
    for most of the classifications. The alpha can be set less
    conservative because this criteria is more robust against
    statistical fluctuationsc.

    Parameters
    ----------
    clf: list
        List of trained classifier.

    ratio : float, optional (default=0.9)
        Ratio of classifiers in which the feature should be removed.

    alpha : float, optional (default=0.05)
        Parameter tuning the threshold. See feature_importance_mad
        describtion.

    Returns
    -------
    kept: numpy.boolarray, shape=(n_features)
        Whether the feature passes the MAD criteria.

    feature_importance: numpy.array, shape=(n_features)
        Array of the importance values for the features. If a list of
        classifier is passed, it is the mean over all classifier.

    feature_importance_std: numpy.array, shape=(n_features)
        If a list of classifier is passed the standard deviation is of
        the feature importance values is returned. Otherwise None is
        returned
    """
    desired_characteristics = ClassifierCharacteristics()
    desired_characteristics.opts['has:feature_importances_'] = True
    assert isinstance(clfs, list), 'List of classifier has to be provided'
    kept_arr = []
    feature_importances = []
    for i, clf_i in enumerate(clfs):
        kept, feature_importance, _ = feature_importance_mad(clf_i,
                                                             alpha=alpha)
        kept_arr.append(kept)
        feature_importances.append(feature_importance)
    kept_arr = np.array(kept_arr)
    feature_importances = np.array(feature_importances)
    feature_importance = np.mean(feature_importances, axis=0)
    feature_importance_std = np.std(feature_importances, axis=0, ddof=1)
    kept = np.sum(kept_arr, axis=0) >= ratio * kept_arr.shape[0]
    return kept, feature_importance, feature_importance_std


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

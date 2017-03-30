# -*- coding:utf-8 -*-
import numpy as np

from ..basics.classifier_characteristics import ClassifierCharacteristics


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


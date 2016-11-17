#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold
    old_kfold = False
except ImportError:
    from sklearn.cross_validation import StratifiedKFold
    old_kfold = True
from sklearn.metrics import roc_auc_score


def __single_auc_score__(feature_i,
                         clf,
                         cv_indices,
                         X,
                         y,
                         sample_weight=None):
    """Method determining the 'area under curve' for a single test set.
    This function is intended for internal use.
    Parameters
    ----------
    feature_i: int
        Index of the tested feature.

    clf: object
        Classifier that should be used for the classification.
        It needs a fit and a predict_proba function.

    cv_indices: list of tuples
        Indices for all the cross validation steps. They are explicit
        pass, so all test sets use the same splitting.

    X : numpy.float32array, shape=(n_samples, n_obs)
        Values describing the samples.

    y : numpy.float32array, shape=(n_samples)
        Array of the true labels.

    sample_weight : None or numpy.float32array, shape=(n_samples)
        If weights are used this has to contain the sample weights.
        None in the case of no weights.

    Returns
    -------
    feature_i: int
        Index of the tested feature. It is need as a return value for
        asynchronous parallel processing

    auc_score: float
        Returns calculated auc score.
    """
    y_pred = np.zeros_like(y, dtype=float)
    for i, [train_idx, test_idx] in enumerate(cv_indices):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        if sample_weight is None:
            sample_weight_train = None
            sample_weight_test = None
        else:
            sample_weight_train = sample_weight[train_idx]
            sample_weight_test = sample_weight[test_idx]
        clf = clf.fit(X=X_train,
                      y=y_train,
                      sample_weight=sample_weight_train)
    y_pred[test_idx] = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y, y_pred, sample_weight=sample_weight_test)
    return feature_i, auc_score


def get_all_auc_scores(clf,
                       selected_features,
                       X,
                       y,
                       sample_weight=None,
                       cv_steps=10,
                       n_jobs=1,
                       forward=True):
    """Method determining the 'area under curve' for all not yet
    selected features. In this function also the feature sets for the
    tests are created.
    Parameters
    ----------
    clf: object
        Classifier that should be used for the classification.
        It needs a fit and a predict_proba function.

    selected_features: list of ints
        List of already selected features

    X : numpy.float32array, shape=(n_samples, n_obs)
        Values describing the samples.

    y : numpy.float32array, shape=(n_samples)
        Array of the true labels.

    sample_weight : None or numpy.float32array, shape=(n_samples)
        If weights are used this has to contains the sample weights.
        None in the case of no weights.

    n_jobs: int, optional (default=1)
        Number of parallel jobs spawned in each a classification in run.
        Total number of used cores is the product of n_jobs from the clf
        and the n_jobs of this function.

    forward: bool, optional (default=True)
        If True it is a 'forward selection'. If False it is a 'backward
        elimination'.

    Returns
    -------
    auc_scores: np.array float shape(n_features_total)
        Return a array containing the auc values. np.nan is the feature
        is already selected.
    """
    selected_features = np.array(selected_features, dtype=int)
    if cv_steps < 2:
        raise ValueError('\'cv_steps\' must be 2 or higher')
    else:
        if old_kfold:
            cv_iterator = StratifiedKFold(y, n_folds=cv_steps,
                                          shuffle=True)
            cv_indices = [[train, test] for train, test in cv_iterator]
        else:
            strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                          shuffle=True)
            cv_iterator = strat_kfold.split(X, y)
            cv_indices = [[train, test] for train, test in cv_iterator]
    test_features = np.array([int(i) for i in range(X.shape[1])
                              if i not in selected_features], dtype=int)

    process_args = []
    for feature_i in test_features:
        if forward:
            set_i = np.hstack((selected_features, feature_i))
            test_set = np.sort(set_i)
        else:
            set_i = list(test_features)
            set_i.remove(feature_i)
            test_set = np.array(set_i)
        process_args.append([feature_i, X[:, test_set],
                             y,
                             sample_weight,
                             clf])

    test_sets = {}
    for feature_i in test_features:
        if forward:
            set_i = np.hstack((selected_features, feature_i))
            test_sets[feature_i] = np.sort(set_i)
        else:
            set_i = list(test_features)
            set_i.remove(feature_i)
            test_sets[feature_i] = np.array(set_i)

    auc_scores = np.empty(X.shape[1])
    auc_scores[:] = np.nan
    if n_jobs > 1:
        futures = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for feature_i, test_set in test_sets.items():
                futures.append(executor.submit(__single_auc_score__,
                                               feature_i=feature_i,
                                               clf=clf,
                                               cv_indices=cv_indices,
                                               X=X[:, test_set],
                                               y=y,
                                               sample_weight=sample_weight))
        results = wait(futures)
        for future_i in results.done:
            feature_i, auc = future_i.result()
            auc_scores[feature_i] = auc
    else:
        auc_scores = []
        for feature_i, test_set in test_sets.items():
            _, auc = __single_auc_score__(feature_i=feature_i,
                                                  clf=clf,
                                                  cv_indices=cv_indices,
                                                  X=X[:, test_set],
                                                  y=y,
                                                  sample_weight=sample_weight)
            auc_scores[feature_i] = auc
    return auc_scores

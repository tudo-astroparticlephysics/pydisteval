#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, wait

try:
    from sklearn.model_selection import StratifiedKFold
    old_kfold = False
except ImportError:
    from sklearn.cross_validation import StratifiedKFold
    old_kfold = True
from sklearn.metrics import roc_auc_score


def get_single_auc_score(feature_i,
                         clf,
                         cv_indices,
                         X,
                         y,
                         sample_weights=None):
    y_pred = np.zeros_like(y, dytpe=float)
    for i, [train_idx, test_idx] in enumerate(cv_indices):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        if sample_weight is None:
            sample_weight_train = None
            sample_weight_test = None
        else:
            sample_weight_train = sample_weight[train_idx]
            sample_weight_test = sample_weight[test_idx]
        clf_i = clf_i.fit(X=X_train,
                      y=y_train,
                      sample_weight=sample_weight_train)
    y_pred[test_idx] = clf_i.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred, sample_weight=sample_weight_test)
    return [feature_i, auc]


def get_all_auc_scores(clf,
                       selected_features,
                       X,
                       y,
                       sample_weight=None,
                       cv_steps=10,
                       n_jobs=1,
                       forward=True):
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
    test_features = np.array([i for i in range(X.shape[1])
                              if i not in selected_features])

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
                             sample_weights,
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

    if n_jobs > 1:
        futures = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for feature_i, test_set in test_sets.items()
                futures.append(executor.submit(get_single_auc_score,
                                               feature_i=feature_i,
                                               clf=clf,
                                               cv_indices=cv_indices,
                                               X=X[:, test_set],
                                               y=y,
                                               sample_weight=sample_weight))
            result = wait(futures)
        results = [future_i.result for future_i in wait(futures).done]
    else:
        results = []
        for feature_i, test_set in test_sets.items():
            results.append(get_single_auc_score(feature_i=feature_i,
                                                clf=clf,
                                                cv_indices=cv_indices,
                                                X=X[:, test_set],
                                                y=y,
                                                sample_weight=sample_weight))
    return results



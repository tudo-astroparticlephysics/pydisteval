#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    from sklearn.model_selection import StratifiedKFold
    old_kfold = False 
except ImportError:
    from sklearn.cross_validation import StratifiedKFold
    old_kfold = True
from sklearn.metrics import roc_curve, auc

from .scripts.preparation import prepare_data, ClassifierCharacteristics

__author__ = "Mathis Börner and Jens Buß"


def main():
    """Entry point for the application script"""
    print("Call your main application code here")


def roc_mismatch(test_df,
                 ref_df,
                 clf,
                 cv_steps=10,
                 test_weight=None,
                 ref_weight=None,
                 test_ref_ratio=1.):
    """Runs a classification betwenn the test data and the reference data.
    For this classification the ROC-Curve  is analysed to check if the
    classifier is sensitive for potential mismathces.
    The hypothesis for the analyse is that the test data has the same
    distribtuion as the reference data.

    Parameters
    ----------
    test_df : pandas.Dataframe, shape=(n_samples_mc, features)
        Dataframe of the test data

    ref_df : pandas.Dataframe, shape=(n_samples_mc, features)
        Dataframe of the reference data

    test_weight : str or None, optional (default=None)
        Name of the columns containing the sample weight of the test
        data. If None no weights will be used.

    ref_weight : str or None, optional (default=None)
        Name of the columns containing the sample weight of the
        reference data. If None no weights will be used.

    test_ref_ratio: float, optional (default=1.)
        Ratio of test and train data. If weights are provided, the ratio
        is for the sum of weights.

    Returns
    -------
    ???
    """
    desired_characteristics = ClassifierCharacteristics()
    desired_characteristics.opts['callable:fit'] = True
    desired_characteristics.opts['callable:predict_proba'] = True
    clf_characteristics = ClassifierCharacteristics(clf)
    assert clf_characteristics.fulfilling(desired_characteristics), \
        'Classifier sanity check failed!'
    X, y, sample_weight, obs = prepare_data(test_df,
                                            ref_df,
                                            test_weight=None,
                                            ref_weight=None,
                                            test_ref_ratio=1.)

    if old_kfold:
        cv_iterator = StratifiedKFold(y, n_folds=cv_steps,
                                      shuffle=True)
    else:
        strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                      shuffle=True)
        cv_iterator = strat_kfold.split(X, y)
    y_pred = np.zeros_like(y, dtype=float)
    roc_curves = []

    for train_idx, test_idx in cv_iterator:
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
        clf = clf.fit(X_train, y_train, sample_weight_train)
        y_pred_test = clf.predict_proba(X_test)[:, 1]
        roc_curves.append(roc_curve(y_test, y_pred_test,
                                    sample_weight=sample_weight_test))
        y_pred[test_idx] = y_pred_test

    auc_values = [auc(fpr, tpr) for fpr, tpr, _ in roc_curves]

    print('AUC: %.3f +/- %.3f' % (np.mean(auc_values), np.std(auc_values)))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from tqdm import tqdm

from logging import getLogger

try:
    from sklearn.model_selection import StratifiedKFold
    old_kfold = False
except ImportError:
    from sklearn.cross_validation import StratifiedKFold
    old_kfold = True
from sklearn.metrics import roc_curve, auc

from .scripts import ClassifierCharacteristics
from .scripts import prepare_data

logger = getLogger('disteval')

__author__ = "Mathis Börner and Jens Buß"

def roc_mismatch(clf,
                 X,
                 y,
                 sample_weight=None,
                 cv_steps=10):
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

    if old_kfold:
        cv_iterator = StratifiedKFold(y, n_folds=cv_steps,
                                      shuffle=True)
    else:
        strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                      shuffle=True)
        cv_iterator = strat_kfold.split(X, y)
    y_pred = np.zeros_like(y, dtype=float)
    cv_step = np.zeros_like(y, dtype=int)

    for i, [train_idx, test_idx] in tqdm(enumerate(cv_iterator)):
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
        clf = clf.fit(X=X_train,
                      y=y_train,
                      sample_weight=sample_weight_train)
        y_pred[test_idx] = clf.predict_proba(X_test)[:, 1]
        cv_step[test_idx] = i
    return y_pred, cv_step, clf

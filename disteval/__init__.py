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

<<<<<<< HEAD
from .scripts.classifier_characteristics import ClassifierCharacteristics
from .scripts.preparation import prepare_data

__author__ = "Mathis Börner and Jens Buß"
=======
from .scripts import ClassifierCharacteristics
from .scripts import prepare_data
>>>>>>> 84d55a86e8890533765e630e4ee6153fec442a9b

logger = getLogger('disteval')

__author__ = "Mathis Börner and Jens Buß"

def cv_test_ref_classification(clf,
                 X,
                 y,
                 sample_weight=None,
                 cv_steps=10,
                 return_all_models=False):
    """Runs a classification betwenn the test data and the reference data.
    This classification is run in a cross-validation with a provided
    classifier. The classifier needs a fit function to start the model
    building process and a predict_func to obtain the classifier score.
    The score is expected to be between 0 and 1.

    Parameters
    ----------
    clf: object
        Classifier that should be used for the classification.
        It needs a fit and a predict_proba function.

    X : numpy.float32array, shape=(n_samples, n_obs)
        Values describing the samples.

    y : numpy.float32array, shape=(n_samples)
        Array of the true labels.

    sample_weight : None or numpy.float32array, shape=(n_samples)
        If weights are used this has to contains the sample weights.
        None in the case of no weights.

    cv_steps: int, optional (default=10)
        Number of cross-validation steps. If < 2 the model is trained on
        all samples and no prediction is made.

    return_all_models: bool, optional (default=False)
        If all models for the cross-validiation should be saved and 
        returned.

    Returns
    -------
    clf: object
        Trained classifier. If return_all_models, a liste of all trained
        classifiers, is returned.

    y_pred : numpy.float32array, shape=(n_samples)
        Array of the classifier score.

    cv_step : numpy.int, shape=(n_samples)
        Iteration in which the sample was classified.
    """
    desired_characteristics = ClassifierCharacteristics()
    desired_characteristics.opts['callable:fit'] = True
    desired_characteristics.opts['callable:predict_proba'] = True

    clf_characteristics = ClassifierCharacteristics(clf)
    assert clf_characteristics.fulfilling(desired_characteristics), \
        'Classifier sanity check failed!'

    if cv_steps < 2:
        clf = clf.fit(X=X,
                      y=y,
                      sample_weight=sample_weight)
        return clf, None, None

    else:
        if old_kfold:
            cv_iterator = StratifiedKFold(y, n_folds=cv_steps,
                                          shuffle=True)
        else:
            strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                          shuffle=True)
            cv_iterator = strat_kfold.split(X, y)
        y_pred = np.zeros_like(y, dtype=float)
        cv_step = np.zeros_like(y, dtype=int)
        if return_all_models:
            from copy import deepcopy
            trained_clfs= []
        for i, [train_idx, test_idx] in enumerate(cv_iterator):
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
            if return_all_models:
                trained_clfs.append(deepcopy(clf))
        if return_all_models:
            clf = trained_clfs
        return clf, y_pred, cv_step

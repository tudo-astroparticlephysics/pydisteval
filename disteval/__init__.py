#!/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from sklearn.model_selection import StratifiedKFold

from .scripts.classifier_characteristics import ClassifierCharacteristics
from .scripts.recursive_selection_parallel import get_all_auc_scores


logger = getLogger('disteval')

__author__ = "Mathis Börner and Jens Buß"


def cv_test_ref_classification(clf,
                               X,
                               y,
                               sample_weight=None,
                               cv_steps=10,
                               return_all_models=False,
                               random_state=None):
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

    random_state: None, int or RandomState
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

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
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
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
        strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                      shuffle=True,
                                      random_state=random_state)
        cv_iterator = strat_kfold.split(X, y)
        y_pred = np.zeros_like(y, dtype=float)
        cv_step = np.zeros_like(y, dtype=int)
        if return_all_models:
            from copy import deepcopy
            trained_clfs = []
        for i, [train_idx, test_idx] in enumerate(cv_iterator):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            if sample_weight is None:
                sample_weight_train = None
            else:
                sample_weight_train = sample_weight[train_idx]
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


def recursive_feature_selection_roc_auc(clf,
                                        X,
                                        y,
                                        sample_weight=None,
                                        n_features=10,
                                        cv_steps=10,
                                        n_jobs=1,
                                        forward=True,
                                        matching_features=True):
    """Method building a feature set in a recursive fashion. Depending
    on the setting it is run as a forward selection/backward elimination
    searching for a set of n features with the highest/lowest mismatch.
    To get the set with the size n starting from n_total features the
    following approaches are used:

    Forward Selection:
    To get the k+1 set every not yet selected feature is used to
    generate (n_total - k sets). The set with the best score is the
    k + 1 set. Those steps are repeated until n features are selected

    Backward Elimination:
    To get k+1 eliminated features every not yet eleminated feature is used
    to generate (n_total - k) sets. The sets consist of all not yet
    eliminated features minus the one that is tested. The set with the
    best score determines the next feature to eliminate. Those steps are
    repeated until n features are eliminated.

    What the best score depends also on the settings:
    matching_features:
        forward: min(|auc - 0.5|)
        not forward: max(|aux - 0.5|)

    not matching_features:
        forward: max(auc )
        not forward: min(aux)


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

    n_features : int, optional (default=10)
        Number of feature that are selected (forward=True) or eliminated
        (forward=False)

    n_jobs: int, optional (default=1)
        Number of parallel jobs spawned in each a classification in run.
        Total number of used cores is the product of n_jobs from the clf
        and the n_jobs of this function.

    forward: bool, optional (default=True)
        If True it is a 'forward selection'. If False it is a 'backward
        elimination'.

    matching_features: bool, optional (default=True)
        Wether for matching or mismatching feature should be searched

    Returns
    -------
    selected_features: list of ints
        Return a list containing the indeces of X, that were
        selected/eliminated. The order corresponds to the order the
        features were selected/eliminated.

    auc_scores: np.array float shape(n_features_total, n_features)
        Return a array containing the auc values for all steps.
        np.nan is the feature was already selected in the specific run.
    """
    desired_characteristics = ClassifierCharacteristics()
    desired_characteristics.opts['callable:fit'] = True
    desired_characteristics.opts['callable:predict_proba'] = True

    clf_characteristics = ClassifierCharacteristics(clf)
    assert clf_characteristics.fulfilling(desired_characteristics), \
        'Classifier sanity check failed!'

    if n_features > X.shape[1]:
        logger.info(' \'n_features\' higher than total number of features.'
                    ' \'n_features\' reduced!')
        n_features = X.shape[1]
    auc_scores = np.zeros((X.shape[1], n_features))
    selected_features = []

    while len(selected_features) != n_features:
        auc_scores_i = get_all_auc_scores(clf,
                                          selected_features,
                                          X,
                                          y,
                                          sample_weight=sample_weight,
                                          cv_steps=cv_steps,
                                          n_jobs=n_jobs,
                                          forward=forward)
        value_best = None
        index_best = None
        for idx, auc in enumerate(auc_scores_i):
            if not np.isfinite(auc):
                continue
            if value_best is None:
                value_best = auc
                index_best = idx
            if matching_features:
                if forward:
                    if np.abs(auc - 0.5) < np.abs(value_best - 0.5):
                        value_best = auc
                        index_best = idx
                else:
                    if np.abs(auc - 0.5) > np.abs(value_best - 0.5):
                        value_best = auc
                        index_best = idx
            else:
                if forward:
                    if auc > value_best:
                        value_best = auc
                        index_best = idx
                else:
                    if auc < value_best:
                        value_best = auc
                        index_best = idx
        auc_scores[:, len(selected_features)] = auc_scores_i
        selected_features.append(index_best)
    return selected_features, auc_scores

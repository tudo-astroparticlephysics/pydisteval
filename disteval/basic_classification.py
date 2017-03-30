# -*- coding: utf-8 -*-
from logging import getLogger

import numpy as np

from sklearn.model_selection import StratifiedKFold

from .basics.classifier_characteristics import ClassifierCharacteristics

logger = getLogger('disteval.basic_classification')


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
        logger.info('cv_steps were < 2, so the classifier was trained with'
                    ' all provided data!')
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

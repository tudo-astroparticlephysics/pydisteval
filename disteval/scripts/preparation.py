#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np


def prepare_data(test_df,
                 ref_df,
                 test_weight=None,
                 ref_weight=None,
                 test_ref_ratio=1.):
    """Makes the data usable for sklearn.

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
    X : numpy.float32array, shape=(n_samples, n_obs)
        Values of the columns which appeared in both Dataframes and
        are not used as Weights

    y : numpy.float32array, shape=(n_samples)
        Array of the true labels.
            1 = Reference
            0 = Test

    sample_weight : None or numpy.float32array, shape=(n_samples)
        Not None if ref_weight and/or test_weight was provided. If array
        is returned, it contains the sample weights

    obs : list[str]
        List of the names of the columns of X
    """
    test_obs = set(test_df.columns)
    ref_obs = set(ref_df.columns)
    use_weights = False
    if test_weight is not None:
        use_weights = True
        try:
            test_obs.remove(test_weight)
        except KeyError:
            raise KeyError('Weight \'%s\' not in test dataframe')
        else:
            sample_weight_test = np.array(test_df.loc[:, test_weight].values,
                                         dtype=np.float32)
    elif test_weight is not None:
        sample_weight_test = np.ones(len(test_obs), dtype=np.float32)
    if ref_weight is not None:
        use_weights = True
        try:
            ref_obs.remove(ref_weight)
        except KeyError:
            raise KeyError('Weight \'%s\' not in reference dataframe')
        else:
            sample_weight_ref = np.array(ref_df.loc[:, ref_weight].values,
                                         dtype=np.float32)
    elif test_weight is not None:
        sample_weight_ref = np.ones(len(ref_df), dtype=np.float32)

    if len(set.difference(ref_obs, test_obs)) > 0:
        unique_obs_test = test_obs.difference(ref_obs)
        unique_obs_ref = test_obs.difference(ref_obs)
        msg = 'Dataset are not consistent: '
        for o in unique_obs_ref:
            msg += ' ref.%s' % o
        for o in unique_obs_test:
            msg += ' test.%s' % o
        msg += ' will be ignored'
        warnings.warn(msg)


    obs = set.intersection(mc_obs, data_obs)

    test_df = test_df.loc[:, obs]
    ref_df = ref_df.loc[:, obs]
    X_test = np.array(test_df.loc[:, obs].values, dtype=np.float32)
    X_ref = np.array(ref_df.loc[:, obs].values, dtype=np.float32)
    y_test = np.zeros(X_test.shape[0], dtype=int)
    y_ref = np.ones(X_ref.shape[0], dtype=int)

    isfinite_test = np.isfinite(X_test)
    selected = np.sum(isfinite_test, axis=1) == len(obs)
    print(selected)
    n_selected = np.sum(selected)
    if n_selected < X_test.shape[0]:
        n_removed = X_test.shape[0] - n_selected
        warnings.warn('%d NaNs removed from the test data' % n_removed)
    X_test = X_test[selected, :]
    y_test =  y_test[selected]
    if use_weights:
        sample_weight_test = sample_weight_test[selected, :]

    isfinite_ref = np.isfinite(X_ref)
    selected = np.sum(isfinite_ref, axis=1) == len(obs)
    n_selected = np.sum(selected)
    if n_selected < X_ref.shape[0]:
        n_removed = X_ref.shape[0] - n_selected
        warnings.warn('%d NaNs removed from the ref data' % n_removed)
    X_ref = X_ref[selected, :]
    y_ref =  y_ref[selected]
    if use_weights:
        sample_weight_ref = sample_weight_ref[selected, :]

    if use_weights:
        sum_w_test = np.sum(sample_weight_test)
        sum_w_ref = np.sum(sample_weight_ref)
        if sum_w_test / sum_w_ref > test_ref_ratio:
            probability = (test_ref_ratio * sum_w_ref) / sum_w_test
            seleceted = np.random.uniform(size=X_test.shape[0]) <= probability
            X_test = X_test[seleceted, :]
            y_test = y_test[seleceted]
            sample_weight_test = sample_weight_test[seleceted, :]
        elif sum_w_test / sum_w_ref <= test_ref_ratio:
            probability = sum_w_test / (sum_w_ref * test_ref_ratio)
            seleceted = np.random.uniform(size=y_ref.shape[0]) <= probability
            X_ref = X_ref[seleceted, :]
            y_ref = y_ref[seleceted]
            sample_weight_ref = sample_weight_ref[seleceted, :]
        X = np.vstack((X_test, X_ref))
        y = np.hstack((y_test, y_ref))
        sample_weight = np.vstack((sample_weight_test, sample_weight_ref))
    else:
        n_rows_test = len(y_test)
        n_rows_ref = len(y_ref)
        if n_rows_test / n_rows_ref > test_ref_ratio:
            probability = (test_ref_ratio * n_rows_ref) / n_rows_test
            seleceted = np.random.uniform(size=X_test.shape[0]) <= probability
            X_test = X_test[seleceted, :]
            y_test = y_test[seleceted]
        elif n_rows_test / n_rows_ref > test_ref_ratio:
            probability = n_rows_ref / (n_rows_test * test_ref_ratio)
            seleceted = np.random.uniform(size=y_ref.shape[0]) <= probability
            X_ref = X_ref[seleceted, :]
            y_ref = y_ref[seleceted]
        X = np.vstack((X_test, X_ref))
        y = np.hstack((y_test, y_ref))
        sample_weight = None
    return X, y, sample_weight, obs


class ClassifierCharacteristics(object):
    """Class to define and compare Characteristics of classifier.
    The core of the Class is the dict ops containing keys whether
    attributes or functions are required/forbidden. Keys like
    'callable_fit' are True if the classifier has a callable function
    'fit'. Keys like 'has_feature_importance' are True if the classifier
    has an atribute 'feature_importance'.
    True in the dict means function/attribute is needed or present.
    False means function/attribute is forbidden or not present.
    None in the dict means is ignore in the evaluation

    Parameters
    ----------
    clf: None or object
        If None the dict is initiated with None for all keys.
        If clf is provided the dict is contains only True and False
        depending on the clf characteristics

    Attributes
    ----------
    opts : dict
        Dictionary containing all the needed/desired characteristics."""
    def __init__(self, clf=None):
        self.opts = {
            'callable_fit': None,
            'callable_predict': None,
            'callable_predict_proba': None,
            'callable_decision_function': None,
            'has_feature_importance': None}
        if clf is not None:
            for key in self.opts.keys():
                self.opts[key] = False
                if key.startswith('callable_'):
                    desired_callable = key.replace('callable_', '')
                    if hasattr(clf, desired_callable):
                        if callable(clf.desired_callable):
                            self.opts[key] = True
                if key.startswith('has_'):
                    desired_attribute = key.replace('has_', '')
                    if hasattr(clf, desired_callable):
                        self.opts[key] = True

    def __eq__(self, second_instance):
        check_keys = [k for k, v in second_instance.opts.items()
                      if v is not None]
        for key in check_keys:
            if self.opts[key] != second_instance.opts[key]:
                if self.opts[k]:
                    msg = 'Provided classifier has %s' % key
                else:
                    msg = 'Provided classifier is missing %s' % key
                raise AttributeError(msg)
        return True


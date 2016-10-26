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
    'callable:fit' are True if the classifier has a callable function
    'fit'. Keys like 'has:feature_importance' are True if the classifier
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
        Dictionary containing all the needed/desired characteristics.

    clf : object
        If a clf is provided, a pointer to the classifier is stored.
        To check characteristics later on."""
    def __init__(self, clf=None):
        self.opts = {
            'callable:fit': None,
            'callable:predict': None,
            'callable:predict_proba': None,
            'callable:decision_function': None,
            'has:feature_importance': None}
        if clf is not None:
            self.clf = clf
            for key in self.opts.keys():
                self.opts[key] = self.__evalute_clf__(key)

    def __evalute_clf__(self, key):
        """Check if the classifier provides the attribute/funtions
        asked for with the key. Keys must start with either "callable:"
        or "has:".
        "callable:<name>"  would check for a funtions with the name <name>.
        "has:<name>"  would check for a attribute with the name <name>.
        Parameters
        ----------
        key: str
            If None the dict is initiated with None for all keys.
            If clf is provided the dict is contains only True and False
            depending on the clf characteristics

        Returns
        ----------
        present : bool
            Boolean whether the asked for characteristic is present"""
        if key.startswith('callable:'):
            desired_callable = key.replace('callable:', '')
            if hasattr(clf, desired_callable):
                if callable(getattr(clf, desired_callable)):
                    return True
        elif key.startswith('has:'):
            desired_attribute = key.replace('has:', '')
            if hasattr(clf, desired_attribute):
                return True
        else:
            print(key)
            raise ValueError('Opts keys have to start with eiter callable:'
                             ' for functions or has: for attributes')
        return False

    def fulfilling(self, second_instance, two_sided=False):
        """Check if the classifier provides the attribute/funtions
        asked for with the key. Keys must start with either "callable:"
        or "has:".
        "callable:<name>"  would check for a funtions with the name <name>.
        "has:<name>"  would check for a attribute with the name <name>.
        Parameters
        ----------
        second_instance: ClassifierCharacteristics
            Second instance of a ClassifierCharacteristics which defines
            the needed characteristics.

        two_sided: boolean, optional (default=False)
            If False only the characteristics asked for in the second
            instance has to be fulfilled. If two_sided is True. Both
            instances has to be the same (equivalent to __eq__)

        Returns
        ----------
        present : bool
            Boolean whether the asked for characteristic is present"""
        if two_sided:
            check_keys_1 = set([k for k, v in self.opts.items()
                                if v is not None])
            check_keys_2 = set([k for k, v in second_instance.opts.items()
                                if v is not None])
            check_keys = check_keys_1.intersection(check_keys_2)
        else:
            check_keys = [k for k, v in second_instance.opts.items()
                          if v is not None]
        for key in check_keys:
            if key not in self.opts.keys():
                if hasattr(self, clf):
                    value = self.__evalute_clf__(key)
                    self.opts[key] = value
                else:
                    raise KeyError('%s not set for the comparison partner')
            if key not in second_instance.opts.keys():
                if hasattr(second_instance, clf):
                    value = second_instance.__evalute_clf__(key)
                    second_instance.opts[key] = value
                else:
                    raise KeyError('%s not set for the comparison partner')
            if self.opts[key] != second_instance.opts[key]:
                att = key.replace('callable:', '')
                att = att.replace('has:', '')
                if self.opts[key]:
                    msg = 'Provided classifier has %s' % att
                else:
                    msg = 'Provided classifier is missing %s' % att
                raise AttributeError(msg)
        return True

    def __eq__(self, second_instance):
        return self.fulfilling(second_instance, two_sided=True)

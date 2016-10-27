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
    test_df : pandas.Dataframe, shape=(n_samples, X_names)
        Dataframe of the test data

    ref_df : pandas.Dataframe, shape=(n_samples, X_names)
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

    X_names : list[str]
        List of the names of the columns of X
    """
    # make the dataframe homogenious
    test_X_names = set(test_df.columns)
    ref_X_names = set(ref_df.columns)
    # Check if weights are used
    use_weights = False
    if test_weight is not None:
        use_weights = True
        try:
            test_X_names.remove(test_weight)
        except KeyError:
            raise KeyError('Weight \'%s\' not in test dataframe')
        else:
            # convert to float32 numpy array for sklearn
            sample_weight_test = np.array(test_df.loc[:, test_weight].values,
                                          dtype=np.float32)
    elif test_weight is not None:
        # If ref uses weights, dummy weights are created
        sample_weight_test = np.ones(len(test_X_names), dtype=np.float32)
    else:
        sample_weight_test = None
    if ref_weight is not None:
        use_weights = True
        try:
            ref_X_names.remove(ref_weight)
        except KeyError:
            raise KeyError('Weight \'%s\' not in reference dataframe')
        else:
            # convert to float32 numpy array for sklearn
            sample_weight_ref = np.array(ref_df.loc[:, ref_weight].values,
                                         dtype=np.float32)
    elif test_weight is not None:
        # If test uses weights, dummy weights are created
        sample_weight_ref = np.ones(len(ref_df), dtype=np.float32)
    else:
        sample_weight_ref = None
    # This sections warns the user about differences between the datasets
    if len(set.difference(ref_X_names, test_X_names)) > 0:
        unique_X_names_test = test_X_names.difference(ref_X_names)
        unique_X_names_ref = ref_X_names.difference(test_X_names)
        msg = 'Dataset are not consistent: '
        for o in unique_X_names_ref:
            msg += ' ref.%s' % o
        for o in unique_X_names_test:
            msg += ' test.%s' % o
        msg += ' will be ignored'
        warnings.warn(msg)

    X_names = set.intersection(test_X_names, ref_X_names)

    test_df = test_df.loc[:, X_names]
    X_test, y_test, sample_weight_test = convert_and_remove_non_finites(
        test_df, sample_weight_test, is_ref=False)

    ref_df = ref_df.loc[:, X_names]
    X_ref, y_ref, sample_weight_ref = convert_and_remove_non_finites(
        ref_df, sample_weight_ref, is_ref=True)

    # In this section the desired test/ref ratio si realized
    if use_weights:
        n_test = np.sum(sample_weight_test)
        n_ref = np.sum(sample_weight_ref)
    else:
        n_test = len(y_test)
        n_ref = len(y_ref)
    if n_test / n_ref > test_ref_ratio:
        probability = (test_ref_ratio * n_ref) / n_test
        seleceted = np.random.uniform(size=y_test.shape[0]) <= probability
        X_test, y_test, sample_weight_test = shrink_data(
            seleceted, X_test, y_test, sample_weight_test)
    elif n_test / n_ref <= test_ref_ratio:
        probability = n_test / (n_ref * test_ref_ratio)
        seleceted = np.random.uniform(size=y_ref.shape[0]) <= probability
        X_ref, y_ref, sample_weight_ref = shrink_data(
            seleceted, X_ref, y_ref, sample_weight_ref)
    # Combining ref and test data into single numpy arrays
    X = np.vstack((X_test, X_ref))
    y = np.hstack((y_test, y_ref))
    if use_weights:
        sample_weight = np.hstack((sample_weight_test, sample_weight_ref))
    else:
        sample_weight = None
    return X, y, sample_weight, X_names


def convert_and_remove_non_finites(df, sample_weight, is_ref=False):
    """Makes the dataframes usable for sklearn.
    For this purpose they are converted to numpy arrays and non finites
    are removed.

    Parameters
    ----------
    df : pandas.Dataframe, shape=(n_samples, n_obs)
        Dataframe that should be converted and filtered

    sample_weight : array-like or None
        Array containing the weights for the samples.

    is_ref : boolean
        Indicates if the provided dataframe is should be treated as
        reference data, so that y is set to 1.

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
    """
    X = np.array(df.values, dtype=np.float32)
    if is_ref:
        y = np.ones(X.shape[0], dtype=int)
        set_name = 'reference set'
    else:
        y = np.zeros(X.shape[0], dtype=int)
        set_name = 'test set'
    isfinite = np.isfinite(X)
    selected = np.sum(isfinite, axis=1) == len(df.columns)
    n_selected = np.sum(selected)
    if n_selected < X.shape[0]:
        n_removed = X.shape[0] - n_selected
        msg = '%d non finites removed from %s' % (n_removed, set_name)
        warnings.warn(msg)
    X = X[selected, :]
    y = y[selected]
    if sample_weight is not None:
        sample_weight = sample_weight[selected, :]
    return X, y, sample_weight


def shrink_data(selected, X, y, sample_weight=None):
    """Shrinks the data arrays by applying the selected mask on X, y
    and the sample weights.

    Parameters
    ----------
    selected : array-like with booleans
        Indicated if a sample should be used or not.

    X : numpy.float32array, shape=(n_samples, n_obs)
        Values describing the samples.

    y : numpy.float32array, shape=(n_samples)
        Array of the true labels.

    sample_weight : None or numpy.float32array, shape=(n_samples)
        If weights are used this array contains the sample weights.

    Returns
    -------
    X : numpy.float32array, shape=(n_samples, n_obs)
        Shrinked values describing the samples.

    y : numpy.float32array, shape=(n_samples)
        Shrinked array of the true labels.

    sample_weight : None or numpy.float32array, shape=(n_samples)
        If weights are used this shrinked array contains the sample weights.
    """
    X = X[selected, :]
    y = y[selected]
    if sample_weight is not None:
        sample_weight = sample_weight[selected, :]
    return X, y, sample_weight

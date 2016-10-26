#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Mathis Börner and Jens Buss"

from sklearn.model_selection import StratifiedKFold

from scripts.preparation import prepare_data, ClassifierCharacteristics


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
    desired_characteristics.opts['callable_fit'] = True
    desired_characteristics.opts['callable_predict_proba'] = True
    clf_characteristics = ClassifierCharacteristics(clf)

    assert ClassifierCharacteristics(clf) == desired_characteristics, \
        'Classifier sanity check failed!'
    X, y, sample_weight, obs = prepare_data(test_df,
                                            ref_df,
                                            test_weight=None,
                                            ref_weight=None,
                                            test_ref_ratio=1.)
    strat_kfold = StratifiedKFold(n_splits=cv_steps,
                                  shuffle=True)
    for train_idx, test_idx in skf.split(X, y):



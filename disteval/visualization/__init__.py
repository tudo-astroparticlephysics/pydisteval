# -*- coding:utf-8 -*-

"""
Collection of methods to visualize the results of disteval functions
"""
from matplotlib import pyplot as plt


def visualize_feature_importance_mad(return_list,
                                     X_names,
                                     save_path=None,
                                     fig_size=(8, 10)):
    """Function visualizing the output of the
    disteval.evaluation.roc_curve_equivalence_ks_test function. It makes
    plots for the ROC curve and the TPR/FPR. It also marks the
    'operation points'.

    Parameters
    ----------
    return_list: list
        List of all returns from the feature_importance_mad or
        feature_importance_mad_majority function.

    X_names: list of strings
        Name of the columns of X.

    save_path : None or string, default=./roc_equivalence.png', optional
        Path under which the plot should be saved. If None only the
        figure and the axes are returned. If 'show' plt.show() is called.

    fig_size : array-like, shape=(2), optional (default=(8,10))

    Returns
    -------
    fig: matplotlib.figure
        The created figure.

    axes: list of matplotlib.axis (len=3)
        The created axes.
    """
    fig = plt.figure(figsize=fig_size)


    if save_path == 'show':
        plt.show()
    elif save_path is not None:
        plt.savefig(save_path)
    return fig, ax

# -*- coding:utf-8 -*-

"""
Collection of methods to visualize the results of disteval functions
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def visualize_feature_importance_mad(return_list,
                                     X_names,
                                     annoting_text='auto',
                                     manual_x_lims=None,
                                     save_path=None,
                                     fig_size=(12, 10)):
    """Function visualizing the output of the
    disteval.evaluation.feature_importance_mad(_majority). It plots
    a histogram for the feature importances with a rug plot.
    Removed features are marked and can be labeled.

    Parameters
    ----------
    return_list: list
        List of all returns from the feature_importance_mad or
        feature_importance_mad_majority function.

    X_names: list of strings
        Name of the columns of X.

    annoting_text: [True, False, 'auto'], optional (default=True)
        Whether the names of the removed features should be plotted.
        If 'auto' the x_lims are autoscaled to try to fit in all the
        names and the names are only printed if 10 or less features
        are removed.

    manual_x_lims: array-like, shape=(2), optional (default=None)
        Array with x limits. Useful if the names of the removed features
        doesn't fit on the figure.

    save_path : None or string, default=./roc_equivalence.png', optional
        Path under which the plot should be saved. If None only the
        figure and the axes are returned. If 'show' plt.show() is called.

    fig_size : array-like, shape=(2), optional (default=(12,10))
        Size of the figure.

    Returns
    -------
    fig: matplotlib.figure
        The created figure.

    axes: list of matplotlib.axis (len=3)
        The created axes.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    kept = return_list[0]
    feature_importance = return_list[1]
    ax.hist(feature_importance, bins=30)
    for x_i in feature_importance[kept]:
        ax.axvline(x_i, 0, 0.05, linewidth=0.3, color='k')

    y_lims = ax.get_ylim()
    x_lims = ax.get_xlim()
    dx = (x_lims[1] - x_lims[0]) * 0.01
    dy = (y_lims[1] - y_lims[0]) * 0.02
    length = (y_lims[1] - y_lims[0]) * 0.08
    y_0 = y_lims[0] + (y_lims[1] - y_lims[0]) * 0.05
    n_removed = sum(~kept)
    if isinstance(annoting_text, bool):
        if annoting_text:
            do_text = True
        else:
            do_text = False
    else:
        if n_removed <= 10:
            do_text = True
            ax.set_xlim(x_lims[0], x_lims[1] + (x_lims[1] - x_lims[0]) * 0.25)
        else:
            do_text = False
    removed_names = [name_i for name_i, kept_i in zip(X_names, kept)
                     if not kept_i]
    removed_x = feature_importance[~kept]
    order = np.argsort(feature_importance[~kept])[::-1]
    for i, idx in enumerate(order):
        x_i = removed_x[idx]
        ax.axvline(x_i, 0, 0.05, linewidth=0.3, color='r', zorder=3)
        if do_text:
            ax.annotate(removed_names[idx],
                        xy=(x_i, y_0),
                        xytext=(x_i + dx, y_0 + length + i * dy),
                        arrowprops=dict(facecolor='0.6',
                                        edgecolor='0.6',
                                        shrink=0.05),
                        size=10,
                        family='monospace',
                        color='0.6')

    ax.set_ylabel('Number of Features')
    ax.set_xlabel('Feature Importance')
    if manual_x_lims is not None:
        ax.set_xlim(manual_x_lims)
    if save_path == 'show':
        plt.show()
    elif save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def visualize_roc_curve_equivalence_test(return_list,
                                         name_a='Reference',
                                         name_b='Test',
                                         save_path=None,
                                         fig_size=(8, 10)):
    """Function visualizing the output of the
    disteval.evaluation.roc_curve_equivalence_ks_test function. It makes
    plots for the ROC curve and the TPR/FPR. It also marks the
    'operation points'.

    Parameters
    ----------
    return_list: list
        List of all returns from the roc_curve_equivalence_ks_test
        function.

    name_a: string, default=('Reference'), optional
        Name of the component passed as 'a' to
        roc_curve_equivalence_ks_test.

    name_b: string, default=('Test'), optional
        Name of the component passed as 'b' to
        roc_curve_equivalence_ks_test.

    save_path : None or string, default=None, optional
        Path under which the plot should be saved. If None only the
        figure and the axes are returned. If 'show' plt.show() is called.

    fig_size : array-like, shape=(2), optional (default=(8,10))
        Size of the figure.

    Returns
    -------
    fig: matplotlib.figure
        The created figure.

    ax: matplotlib.axis
        The created axis.
    """
    op_point_n = return_list[1]
    op_point_p = return_list[2]
    fpr_a_full = return_list[3]
    tpr_a_full = return_list[4]
    fpr_b_full = return_list[5]
    tpr_b_full = return_list[6]
    thresholds = return_list[7]
    fig = plt.figure(figsize=fig_size)

    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.1, right=0.9, top=0.98, bottom=0.40)
    ax1 = plt.subplot(gs1[:, :])

    gs2 = gridspec.GridSpec(2, 1)
    gs2.update(left=0.1, right=0.90, top=0.32, bottom=0.1, hspace=0.15)
    ax3 = plt.subplot(gs2[1, :])
    ax2 = plt.subplot(gs2[0, :])

    ax1.plot(fpr_a_full, tpr_a_full, 'r-', label=name_a)
    ax1.plot(fpr_b_full, tpr_b_full, 'k--', label=name_b)
    ax1.plot(op_point_n[0, :], op_point_n[1, :], 'bo')
    ax1.plot(op_point_p[0, :], op_point_p[1, :], 'gx')
    ax1.set_ylabel('True Positive Rate (TPR)')
    ax1.set_xlabel('False Positive Rate (FPR)')
    ax1.legend(loc=4)

    D_n = np.absolute(fpr_a_full - fpr_b_full)
    D_p = np.absolute(tpr_a_full - tpr_b_full)

    idx_max_n = np.argmax(D_n)
    idx_max_p = np.argmax(D_p)

    ax2.plot([thresholds[idx_max_n], thresholds[idx_max_n]],
             [fpr_a_full[idx_max_n], fpr_b_full[idx_max_n]],
             ls='-',
             color='0.5')
    ax2.plot(thresholds, fpr_a_full, 'r-')
    ax2.plot(thresholds, fpr_b_full, 'k--')

    ax2.plot([thresholds[idx_max_n], thresholds[idx_max_n]],
             [fpr_a_full[idx_max_n], fpr_b_full[idx_max_n]], 'bo')
    ax2.set_ylabel('FPR')
    ax2.set_ylim([0., 1.05])
    ax2.set_xlim([0., 1.])
    ax2.set_xticklabels([])

    ax3.plot([thresholds[idx_max_p], thresholds[idx_max_p]],
             [tpr_a_full[idx_max_p], tpr_b_full[idx_max_p]],
             ls='-',
             color='0.5')
    ax3.plot(thresholds, tpr_a_full, 'r-')
    ax3.plot(thresholds, tpr_b_full, 'k--')

    ax3.plot([thresholds[idx_max_p], thresholds[idx_max_p]],
             [tpr_a_full[idx_max_p], tpr_b_full[idx_max_p]], 'gx')
    ax3.set_ylabel('TPR')
    ax3.set_ylim([0., 1.05])
    ax3.set_xlim([0., 1.])
    ax3.set_xlabel('Threshold')
    if save_path == 'show':
        plt.show()
    elif save_path is not None:
        plt.savefig(save_path)
    return fig, (ax1, ax2, ax3)

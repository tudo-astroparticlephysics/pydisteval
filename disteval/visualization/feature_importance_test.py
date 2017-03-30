# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt


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

# -*- coding:utf-8 -*-

"""
Collection of methods to visualize the results of disteval functions
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec


def visualize_roc_curve_equivalence_test(return_list,
                                         name_a='Reference',
                                         name_b='Test',
                                         save_path='./roc_equivalence.png'):
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

    y_true : numpy.array, shape=(n_samples_a)
        Name of the component passed as 'b' to
        roc_curve_equivalence_ks_test.

    save_path : None or string, default=./roc_equivalence.png', optional
        Path under which the plot should be saved. If None only the
        figure and the axes are returned. If 'show' plt.show() is called.

    Returns
    -------
    fig: matplotlib.figure
        The created figure.

    axes: list of matplotlib.axis (len=3)
        The created axes.
    """

    op_point_n = return_list[1]
    op_point_p = return_list[2]
    fpr_a_full = return_list[3]
    tpr_a_full = return_list[4]
    fpr_b_full = return_list[5]
    tpr_b_full = return_list[6]
    thresholds = return_list[7]
    fig = plt.figure(figsize=(8, 10))

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

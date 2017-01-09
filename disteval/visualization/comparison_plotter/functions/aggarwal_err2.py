#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
import scipy.stats.distributions as sc_dist

from matplotlib.ticker import ScalarFormatter, LogFormatter

def calc_limits(mu, sum_w, sum_w2, alphas=0.68268949, rel=True):
    if isinstance(alphas, float):
        lim_shape = list(mu.shape) + [2]
        lim = np.zeros(lim_shape)
        lower = [slice(None) for _ in range(len(mu.shape))]
        lower.append(0)
        upper = [slice(None) for _ in range(len(mu.shape))]
        upper.append(1)
        lim[lower], lim[upper] = sc_dist.poisson.interval(alphas, mu)
        zero_mask = mu > 0.
        return lim, lim[zero_mask] / mu[zero_mask]

    else:
        lim_shape = list(mu.shape) + [len(alphas), 2]
        mu_shape = mu.shape
        lim_abs = np.zeros(lim_shape)
        lim_rel = np.zeros(lim_shape)
        for i, a in enumerate(alphas):
            lower = [slice(None) for _ in range(len(mu.shape))]
            lower.extend([i, 0])
            upper = [slice(None) for _ in range(len(mu.shape))]
            upper.extend([i, 1])
            zero_mask = mu > 0.
            flat_mu = mu.reshape(np.prod(mu_shape))
            flat_mask = zero_mask.reshape(np.prod(mu_shape))
            lim_lower, lim_upper = sc_dist.poisson.interval(a,
                                                            flat_mu[flat_mask])
            lim_lower_t_rel = np.zeros_like(flat_mask, dtype=float)
            lim_upper_t_rel = np.zeros_like(flat_mask, dtype=float)
            lim_lower_t_abs = np.zeros_like(flat_mask, dtype=float)
            lim_upper_t_abs = np.zeros_like(flat_mask, dtype=float)
            lim_lower_t_rel[flat_mask] = lim_lower / flat_mu[flat_mask]
            lim_upper_t_rel[flat_mask] = lim_upper / flat_mu[flat_mask]
            lim_lower_t_abs[flat_mask] = lim_lower
            lim_upper_t_abs[flat_mask] = lim_upper
            lim_abs[lower] = lim_lower_t_abs.reshape(mu_shape)
            lim_abs[upper] = lim_upper_t_abs.reshape(mu_shape)
            lim_rel[lower] = lim_lower_t_rel.reshape(mu_shape)
            lim_rel[upper] = lim_upper_t_rel.reshape(mu_shape)
        return lim_abs, lim_rel


def calc_p_alpha_bands(ref_hist, sum_w, sum_w2, hist):
    a_ref = sc_dist.poisson.cdf(ref_hist, ref_hist)
    hist_lower = hist[:, :, 0]
    hist_upper = hist[:, :, 1]
    uncert = np.ones_like(hist)
    uncert_lower = uncert[:, :, 0]
    uncert_upper = uncert[:, :, 1]
    for i, [mu, a_mu] in enumerate(zip(ref_hist, a_ref)):
        if mu > 0:
            a_shape = uncert_lower[i].shape
            x_lower = hist_lower[i].reshape(np.prod(a_shape))
            x_upper = hist_upper[i].reshape(np.prod(a_shape))
            a_lower = sc_dist.poisson.cdf(x_lower, mu)
            a_upper = sc_dist.poisson.cdf(x_upper, mu)
            a_lower -= sc_dist.poisson.pmf(x_lower, mu)
            a_upper = (1-a_upper)
            a_lower = (a_lower)
            a_upper /= (1-a_mu)
            a_lower /= a_mu
            a_lower[x_lower == 0] = np.inf
            a_upper[x_upper == 0] = np.inf
            uncert_lower[i] = a_lower.reshape(a_shape)
            uncert_upper[i] = a_upper.reshape(a_shape)
        else:
            a_shape = uncert_lower[i].shape
            x_lower = hist_lower[i].reshape(np.prod(a_shape))
            x_upper = hist_upper[i].reshape(np.prod(a_shape))
            a_lower = np.zeros_like(x_upper)*np.nan
            a_upper = np.zeros_like(x_upper)*np.nan
            a_lower[x_lower > 0] = -np.inf
            a_upper[x_upper > 0] = np.inf
            uncert_lower[i] = a_lower.reshape(a_shape)
            uncert_upper[i] = a_upper.reshape(a_shape)
#    np.isclose() uncert_lower
    uncert[:, :, 0] = uncert_lower * -1
    uncert[:, :, 1] =uncert_upper
    return uncert


def calc_p_alpha_bands_nobs(ref_hist, sum_w, sum_w2, hist):
    # Input expected to be [N_BINS, N_ALPHAS, 2]
    a = np.empty_like(hist)
    for i in range(hist.shape[0]):
        a[i] = calc_p_alpha_bands(ref_hist[i], hist[i])
    return a



def calc_p_alpha_single(ref_hist, sum_w, sum_w2, hist):
    # Input expected to be [N_BINS]
    a_ref = sc_dist.poisson.cdf(ref_hist, ref_hist)
    uncert = np.empty_like(hist)
    for i, [mu, a_mu, x] in enumerate(zip(ref_hist, a_ref, hist)):
        a = sc_dist.poisson.cdf(x, mu)
        if x == 0 and mu == 0:
            uncert[i] = np.NaN
        elif mu == 0:
            uncert[i] = np.inf
        elif x == 0:
            uncert[i] = -np.inf
        elif x > mu:
            uncert[i] = (1-a)/(1-a_mu)
        else:
            a_0 = sc_dist.poisson.pmf(x, mu)
            uncert[i] = (a-a_0)/(-1*a_mu)

    return uncert


def calc_p_alpha_bands_nobs(ref_hist, sum_w, sum_w2, hist):
    # Input expected to be [N_BINS, N_ALPHAS, 2]
    a = np.empty_like(hist)
    for i in range(hist.shape[0]):
        a[i] = calc_p_alpha_bands(ref_hist[i], hist[i])
    return a

def calc_p_alphas_nobs(ref_hist, sum_w, sum_w2, hist):
    # Input expected to be [N_BINS]
    a = np.empty_like(hist)
    for i in range(hist.shape[0]):
        a[i] = calc_p_alpha_single(ref_hist[i], hist[i])
    return a


def map_ratio(y_values, y_min=None, y_0=1.):
    flattened_y = y_values.reshape(np.prod(y_values.shape))
    infinite = np.isinf(flattened_y)
    finite = np.isfinite(flattened_y)
    finite_y = flattened_y[finite]
    finite_y[finite_y > y_0] = np.NaN
    plus_mask = finite_y > 0
    minus_mask = finite_y < 0
    finite_y = np.absolute(finite_y)
    finite_y[plus_mask] = np.log10(finite_y[plus_mask])
    finite_y[minus_mask] = np.log10(finite_y[minus_mask])
    if y_min is None:
        y_min = min(np.min(finite_y[plus_mask]),
                       np.min(finite_y[minus_mask]))
    finite_y /= np.absolute(y_min)
    finite_y[finite_y > 1] = np.inf
    finite_y[finite_y < -1] = -np.inf
    finite_y[minus_mask] *= -1
    tranformed_values = np.zeros_like(flattened_y)
    tranformed_values[:] = np.NaN
    tranformed_values[finite] = finite_y
    tranformed_values[infinite] = flattened_y[infinite]
    return tranformed_values.reshape(y_values.shape)*-1, y_0, y_min


def generate_ticks(y_0, y_min, max_ticks_per_side=5):
    y_min = np.floor(y_min)
    y_0_log = np.log10(y_0)

    tick_pos = []

    n_ticks = 1
    tick_pos.append(y_0_log)
    if y_0_log != np.floor(y_0_log):
        tick_pos.append(np.floor(y_0_log))
        n_ticks += 2
    while tick_pos[-1] > y_min:
        tick_pos.append(tick_pos[-1]-1)
        n_ticks += 2
    n_ticks_per_side = (n_ticks-1)/2
    mayor_step_size = np.ceil(n_ticks_per_side/max_ticks_per_side)
    tick_pos_mapped, _, _ = map_ratio(np.power(10, tick_pos),
                                                y_min=y_min,
                                                y_0=y_0)
    mayor_ticks = []
    mayor_ticks_labels = []

    minor_ticks = []
    minor_ticks_labels = []
    mayor_tick_counter = 0
    for i, [p, l] in enumerate(zip(tick_pos_mapped, tick_pos)):
        lab = 10**l
        lab = u'10$^{\mathregular{%d}}$' % l
        if i == 0:
            mayor_ticks_labels.append(lab)
            mayor_ticks.append(0)
        else:
            if mayor_tick_counter == mayor_step_size:
                mayor_ticks.extend([p*-1, p])
                mayor_ticks_labels.extend([lab, lab])
                mayor_tick_counter = 0
            else:
                minor_ticks.extend([p*-1, p])
                minor_ticks_labels.extend([lab, lab])
                mayor_tick_counter += 1

    return mayor_ticks_labels, mayor_ticks, minor_ticks_labels, minor_ticks






def plot_uncertainties_ratio_mapped(fig,
                             ax_ratio,
                             uncerts_ratio,
                             binning,
                             y_0,
                             y_min,
                             label,
                             color,
                             cmap,
                             alphas):
    n_alphas = len(alphas)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        j = n_alphas - i - 1
        lower_limit = uncerts_ratio[:, j, 0]
        upper_limit = uncerts_ratio[:, j, 1]
        upper_limit[np.isinf(upper_limit)] = 0
        lower_limit[np.isinf(lower_limit)] = -1
        ax_ratio.fill_between(
                             binning,
                             upper_limit,
                             lower_limit,
                             step='pre',
                             color=c)


def plot_uncertainties_ratio(fig,
                             ax_ratio,
                             uncerts_ratio,
                             binning,
                             label,
                             color,
                             cmap,
                             alphas):
    [ax_plus, ax_minus] = ax_ratio
    n_alphas = len(alphas)
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, len(alphas)))
    for i, (c, a) in enumerate(zip(colors[::-1], alphas[::-1])):
        j = n_alphas - i - 1
        lower_limit = np.absolute(uncerts_ratio[:, j, 0])
        upper_limit = uncerts_ratio[:, j, 1]
        upper_limit[np.isinf(upper_limit)] = 1.
        lower_limit[np.isinf(lower_limit)] = 0.
        ax_plus.fill_between(
                             binning,
                             np.ones_like(upper_limit),
                             upper_limit,
                             step='pre',
                             color=c,
                             zorder=1)
        ax_minus.fill_between(
                             binning,
                             np.ones_like(lower_limit),
                             lower_limit,
                             step='pre',
                             color=c,
                             zorder=1)



def plot_data_ratio_mapped(fig,
                           ax,
                           uncerts_ratio,
                           binning,
                           label,
                           color):
    bin_center = binning
    finite_mask = np.isfinite(uncerts_ratio)
    neg_inf = np.isneginf(uncerts_ratio)
    pos_inf = np.isposinf(uncerts_ratio)

    bin_center = bin_center[finite_mask]
    uncerts_ratio = uncerts_ratio[finite_mask]

    markeredgecolor = 'k'
    markerfacecolor = color
    plot_data_ratio_part(fig, ax,
                         bin_center,
                         uncerts_ratio,
                         color,
                         bot=False)
    plot_inf_marker(fig, ax,
                    binning,
                    ~neg_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=True)
    plot_inf_marker(fig, ax,
                    binning,
                    ~pos_inf,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor,
                    bot=False)

def plot_data_ratio_part(fig, ax, x, y, color, bot=True):
    markeredgecolor = 'k'
    markerfacecolor = color
    infinite_mask = np.isinf(y)
    ax.plot(x[~infinite_mask],
            y[~infinite_mask],
            ls='', ms=5,
            mew=1.,
            marker='o',
            markeredgecolor=markeredgecolor,
            markerfacecolor=markerfacecolor,
            zorder=1,
            clip_on=False)

def plot_inf_marker(fig, ax, binning, zero_mask, markeredgecolor='k',
                    markerfacecolor='none', bot=True):
    patches = []
    radius = 0.008
    bbox = ax.get_position()
    x_0 = bbox.x0
    width = bbox.x1 - bbox.x0
    if bot:
        y0 = bbox.y0 + radius
        orientation = np.pi
    else:
        y0 = bbox.y1 - radius
        orientation = 0
    bin_center = (binning[1:] + binning[:-1]) / 2
    binning_width = binning[-1] - binning[0]
    bin_0 = binning[0]
    for bin_i, mask_i in zip(bin_center, zero_mask):
        if not mask_i:
            x_i = ((bin_i - bin_0) / binning_width * width) + x_0
            patches.append(mpatches.RegularPolygon([x_i, y0], 3,
                                                   radius=radius,
                                                   orientation=orientation,
                                                   facecolor=markerfacecolor,
                                                   edgecolor=markeredgecolor,
                                                   transform=fig.transFigure,
                                                   figure=fig,
                                                   linewidth=1.,
                                                   zorder=1))
    fig.patches.extend(patches)








if __name__ == '__main__':
    from matplotlib.ticker import MaxNLocator
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches


    start = 0
    step = 0.25
    end = 30
    mc = np.arange(start, end, step)
    bins = np.arange(start, end+step, step)


    x_data = np.ones_like(bins)
    data = calc_p_alpha_single(bins, None, None, x_data)

    log = False
    if log:
        bins = np.logspace(-4, 2, 100+1)
        mc = np.logspace(-4, 2, 100)

    alphas = [0.682689492, 0.9, 0.99]
    lim_abs, lim_rel = calc_limits(bins, None, None, alphas=alphas, rel=False)
    bands = calc_p_alpha_bands(bins, None, None, lim_abs)

    y_0 = 0.5
    y_min = -5

    mapped_data ,y_0, y_min = map_ratio(data, y_min=y_min, y_0=y_0)
    mapped_bands, y_0, y_min = map_ratio(bands, y_min=y_min, y_0=y_0)

    generate_ticks(y_0, y_min)

    fig = plt.figure(0)
    gs = GridSpec(2, 1)

    ax_plus = fig.add_subplot(gs[0])
    ax_minus = fig.add_subplot(gs[1])
    ax_plus.set_ylim(ymin=1., ymax=1e-5)
    ax_plus.set_yscale("log", nonposy='clip')
    ax_minus.set_ylim(ymin=1e-5, ymax=1.)
    ax_minus.set_yscale("log", nonposy='clip')

    ax_ratio = [ax_plus, ax_minus]

    ax_minus.plot(bins[data < 0], np.absolute(data[data < 0]), 'ow', zorder=1)
    ax_plus.plot(bins[data > 0], np.absolute(data[data > 0]), 'ow', zorder=1)

    gs.update(left=0.1,
              right=0.9,
              top=0.45,
              bottom=0.1,wspace=0., hspace=0.)


    gs2 = GridSpec(1, 1)
    gs2.update(left=0.1,
              right=0.9,
              top=0.9,
              bottom=0.55,wspace=0., hspace=0.)

    ax = fig.add_subplot(gs2[0])



    ax.set_ylim(ymin=-1., ymax=1.)
    print(ax.yaxis.get_major_formatter())
    plot_uncertainties_ratio_mapped(fig,
                             ax,
                             mapped_bands,
                             bins,
                             y_0,
                             y_min,
                             '',
                             'w',
                             'viridis_r',
                             alphas)

    plot_uncertainties_ratio(fig,
                             ax_ratio,
                             bands,
                             bins,
                             '',
                             'w',
                             'viridis_r',
                             alphas)

    plot_data_ratio_mapped(fig,
                           ax,
                           mapped_data,
                           bins,
                           '',
                           'w')

    M_t, M_p, m_t, m_p = generate_ticks(y_0, y_min)



    ax.set_yticks(M_p)
    ax.set_yticklabels(M_t)
    ax.set_yticks(M_p)
    ax.set_yticks(m_p, minor=True)
    ax.yaxis.grid(which='both')
    ax.yaxis.grid(which='minor', alpha=0.2)
    ax.yaxis.grid(which='major', alpha=0.5)
#    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))


    plt.show()






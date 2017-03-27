import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ColorConverter
from colorsys import rgb_to_hls, hls_to_rgb

from .calc_funcs import map_aggarwal_ratio, rescale_limit

from . import legend_entries as le

MAIN_ZORDER = 4


def modify_color(color,
                 d_saturation=0.,
                 d_lightness=0.):
    conv = ColorConverter()
    if not isinstance(color, tuple):
        rgb_color = conv.to_rgb(color)
    else:
        rgb_color = color
    hls_color = rgb_to_hls(*rgb_color)
    new_l = max(0, min(0.9, hls_color[1] + d_lightness))
    new_s = max(0, min(1, hls_color[2] + d_saturation))
    return hls_to_rgb(hls_color[0], new_l, new_s)


def plot_inf_marker(fig,
                    ax,
                    binning,
                    place_marker,
                    markeredgecolor='k',
                    markerfacecolor='none',
                    bot=True,
                    alpha=1.):
    patches = []
    radius = 0.005
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
    for bin_i, place in zip(bin_center, place_marker):
        if place:
            x_i = ((bin_i - bin_0) / binning_width * width) + x_0
            patches.append(mpatches.RegularPolygon(
                [x_i, y0],
                3,
                radius=radius,
                orientation=orientation,
                facecolor=markerfacecolor,
                edgecolor=markeredgecolor,
                transform=fig.transFigure,
                figure=fig,
                linewidth=1.,
                zorder=MAIN_ZORDER + 1,
                alpha=alpha))
    fig.patches.extend(patches)


def plot_finite_marker(ax, x, y, facecolor, edgecolor, alpha):
    ax.plot(x,
            y,
            ls='',
            mew=1.,
            marker='o',
            markeredgecolor=edgecolor,
            markerfacecolor=facecolor,
            alpha=alpha,
            ms='5',
            zorder=MAIN_ZORDER + 1)


def plot_data_style(fig,
                    ax,
                    bin_edges,
                    y,
                    facecolor,
                    edgecolor,
                    alpha,
                    ms='5'):
    zero_mask = y > 0
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2.

    plot_finite_marker(ax,
                       x=bin_mids[zero_mask],
                       y=y[zero_mask],
                       facecolor=facecolor,
                       edgecolor=edgecolor,
                       alpha=alpha)

    plot_inf_marker(fig, ax,
                    bin_edges,
                    ~zero_mask,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    alpha=alpha)
    return le.DataObject(facecolor,
                         edgecolor,
                         facecolor,
                         edgecolor)


def plot_uncertainties(ax, bin_edges, uncert, color, cmap):
    n_alpha = uncert.shape[1]
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.linspace(0.1, 0.9, n_alpha))
    legend_entries = []
    legend_entries.append(le.UncertObject(colors, color))
    for i, c in enumerate(colors[::-1]):
        j = n_alpha - i - 1
        lower_limit = uncert[:, j, 0]
        upper_limit = uncert[:, j, 1]
        mask = np.isfinite(lower_limit)
        lower_limit[~mask] = 0.
        mask = np.isfinite(upper_limit)
        upper_limit[~mask] = 0.
        plot_band(ax,
                  bin_edges,
                  lower_limit,
                  upper_limit,
                  c,
                  alpha=1.,
                  borders=False,
                  brighten=False,
                  zorder=MAIN_ZORDER)
    for i, c in enumerate(colors):
        legend_entries.append(le.UncertObject_single(c))
    return legend_entries


def plot_band(ax,
              bin_edges,
              y_err_low,
              y_err_high,
              color,
              alpha=0.5,
              borders=1.,
              brighten=True,
              zorder=None):
    if isinstance(borders, bool):
        if borders:
            border_lw = 0.3
            plot_borders = True
        else:
            plot_borders = False
    elif isinstance(borders, float):
        border_lw = borders
        plot_borders = True
    else:
        plot_borders = False

    if zorder is None:
        zorder = MAIN_ZORDER - 1
    if brighten:
        band_color = modify_color(color, 0, 0.4)
    else:
        band_color = color
    alpha = min(1., max(0., alpha))
    ax.fill_between(bin_edges,
                    np.append(y_err_low[0], y_err_low),
                    np.append(y_err_high[0], y_err_high),
                    step='pre',
                    color=band_color,
                    edgecolor=band_color,
                    linewidth=0.0,
                    alpha=alpha,
                    zorder=zorder - 1)
    if plot_borders:
        if brighten:
            band_color = modify_color(color, 0, 0.2)
        else:
            band_color = color
        plot_hist(ax,
                  bin_edges,
                  y_err_low,
                  color,
                  lw=border_lw,
                  alpha=1.0,
                  zorder=zorder)
        plot_hist(ax,
                  bin_edges,
                  y_err_high,
                  color,
                  lw=border_lw,
                  alpha=1.0,
                  zorder=zorder)
    # legend_obj = le.
    legend_obj = None
    return legend_obj


def plot_hist(ax,
              bin_edges,
              y,
              color,
              yerr=None,
              lw=1.6,
              alpha=1.0,
              zorder=None):
    if zorder is None:
        zorder = MAIN_ZORDER
    alpha = min(1., max(0., alpha))
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2.
    nan_mask = np.isfinite(y)
    bin_mids_masked = bin_mids[nan_mask]
    y_masked = y[nan_mask]
    xerr_masked = (np.diff(bin_edges) / 2)[nan_mask]
    if yerr is not None:
        yerr_masked = yerr[nan_mask]
    else:
        yerr_masked = None
    errorbar = ax.errorbar(x=bin_mids_masked,
                           y=y_masked,
                           ls='',
                           xerr=xerr_masked,
                           yerr=yerr_masked,
                           color=color,
                           markersize=0,
                           capsize=0,
                           lw=lw,
                           zorder=zorder,
                           label='Test')
    return errorbar


def plot_line(ax,
              bin_edges,
              y,
              color,
              lw=1.6,
              alpha=1.0,
              zorder=None):
    if zorder is None:
        zorder = MAIN_ZORDER
    alpha = min(1., max(0., alpha))
    obj, = ax.plot(bin_edges,
                   np.append(y[0], y),
                   drawstyle='steps-pre',
                   lw=lw,
                   c=color,
                   label='test',
                   alpha=alpha,
                   zorder=zorder)
    return obj


def plot_test_ratio_mapped(fig,
                           ax,
                           bin_edges,
                           ratio,
                           is_above,
                           facecolor,
                           edgecolor,
                           alpha):
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2.
    is_finite = np.isfinite(ratio)
    finite_mask_upper = np.logical_and(is_finite, is_above)
    finite_mask_lower = np.logical_and(is_finite, ~is_above)

    plot_finite_marker(ax,
                       x=bin_mids[finite_mask_upper],
                       y=ratio[finite_mask_upper],
                       facecolor=facecolor,
                       edgecolor=edgecolor,
                       alpha=alpha)
    plot_finite_marker(ax,
                       x=bin_mids[finite_mask_lower],
                       y=ratio[finite_mask_lower],
                       facecolor=facecolor,
                       edgecolor=edgecolor,
                       alpha=alpha)

    oor_mask_upper = np.logical_and(is_above, np.isposinf(ratio))
    no_ratio_mask_upper = np.logical_and(is_above, np.isneginf(ratio))

    plot_inf_marker(fig,
                    ax,
                    bin_edges,
                    oor_mask_upper,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    bot=False)
    plot_inf_marker(fig,
                    ax,
                    bin_edges,
                    no_ratio_mask_upper,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    bot=False,
                    alpha=0.5)

    oor_mask_lower = np.logical_and(~is_above, np.isposinf(ratio))
    no_ratio_mask_lower = np.logical_and(~is_above, np.isneginf(ratio))

    plot_inf_marker(fig,
                    ax,
                    bin_edges,
                    oor_mask_lower,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    bot=True)
    plot_inf_marker(fig,
                    ax,
                    bin_edges,
                    no_ratio_mask_lower,
                    markerfacecolor=facecolor,
                    markeredgecolor=edgecolor,
                    bot=True,
                    alpha=0.5)


def generate_ticks_for_aggarwal_ratio(y_0, y_min, max_ticks_per_side=5):
    y_min = np.floor(y_min)
    y_0_log = np.log10(y_0)

    tick_pos = []

    n_ticks = 1
    tick_pos.append(y_0_log)
    if y_0_log != np.floor(y_0_log):
        tick_pos.append(np.floor(y_0_log))
        n_ticks += 2
    while tick_pos[-1] > y_min:
        tick_pos.append(tick_pos[-1] - 1)
        n_ticks += 2
    n_ticks_per_side = (n_ticks - 1) / 2
    mayor_step_size = np.ceil(n_ticks_per_side / max_ticks_per_side)
    tick_pos_mapped, y_min_ticks = map_aggarwal_ratio(np.power(10, tick_pos),
                                                      y_0=1.)
    tick_pos_mapped = rescale_limit(tick_pos_mapped,
                                    y_min_ticks,
                                    y_min)

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
                mayor_ticks.extend([p * -1, p])
                mayor_ticks_labels.extend([lab, lab])
                mayor_tick_counter = 0
            else:
                minor_ticks.extend([p * -1, p])
                minor_ticks_labels.extend([lab, lab])
                mayor_tick_counter += 1
    return mayor_ticks_labels, mayor_ticks, minor_ticks_labels, minor_ticks

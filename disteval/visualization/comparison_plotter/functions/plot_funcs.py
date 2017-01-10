import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.colors import ColorConverter
from colorsys import rgb_to_hls, hls_to_rgb

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


def plot_inf_marker(fig, ax, binning, zero_mask, markeredgecolor='k',
                    markerfacecolor='none', bot=True, alpha=1.):
    patches = []
    radius = 0.004
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
                zorder=ZORDER+1,
                alpha=alpha))
    fig.patches.extend(patches)

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
                    zorder=zorder-1)
    if plot_borders:
        if brighten:
            band_color = modify_color(color, 0, 0.2)
        else:
            band_color = color
        plot_hist(ax,
                  bin_edges,
                  y_err_low,
                  border_color,
                  lw=border_lw,
                  alpha=1.0,
                  zorder=zorder)
        plot_hist(ax,
                  bin_edges,
                  y_err_high,
                  border_color,
                  lw=border_lw,
                  alpha=1.0,
                  zorder=zorder)

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
    ax.errorbar(x=bin_mids_masked,
                y=y_masked,
                ls='',
                xerr=xerr_masked,
                yerr=yerr_masked,
                color=color,
                markersize=0,
                capsize=0,
                lw=lw,
                zorder=zorder)

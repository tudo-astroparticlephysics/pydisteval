#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


class DataObject(object):
    def __init__(self, fc_t='w', ec_t='k', fc_c='w', ec_c='k', lw=1.):
        self.fc_t = fc_t
        self.ec_t = ec_t
        self.fc_c = fc_c
        self.ec_c = ec_c
        self.lw = lw


class data_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        scale = fontsize / 22
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        radius = height / 2 * scale
        xt_0 = x0 + width - height / 2
        xc_0 = x0 + height / 2 + radius
        yc_0 = y0 + height / 2 * (1 - scale) + radius

        patch_tri = mpatches.RegularPolygon(
            [xt_0, yc_0],
            3,
            radius=radius * 1.5,
            orientation=np.pi,
            facecolor=orig_handle.fc_t,
            edgecolor=orig_handle.ec_t,
            transform=handlebox.get_transform(),
            linewidth=orig_handle.lw)
        patch_circ = mpatches.Circle([xc_0, yc_0], radius,
                                     facecolor=orig_handle.fc_c,
                                     edgecolor=orig_handle.ec_c,
                                     transform=handlebox.get_transform(),
                                     linewidth=orig_handle.lw)
        handlebox.add_artist(patch_tri)
        handlebox.add_artist(patch_circ)
        return patch_circ


class UncertObject(object):
    def __init__(self, colors, linecolor):
        self.colors = colors
        self.linecolor = linecolor


class uncert_handler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        n_alphas = len(orig_handle.colors)
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        x0 = x0 + 0.1
        width, height = handlebox.width, handlebox.height
        y_mid = y0 + height / 2
        step_size = (0.5 * height) / n_alphas
        for i, c in enumerate(orig_handle.colors[::-1]):
            j = n_alphas - i
            y0_i = y_mid - (j * step_size)
            height_i = j * 2 * step_size
            rec = mpatches.Rectangle([x0, y0_i], width,
                                     height_i,
                                     facecolor=c,
                                     edgecolor=c,
                                     transform=handlebox.get_transform(),
                                     zorder=10)
            handlebox.add_artist(rec)
        line = plt.Line2D([x0, x0 + width],
                          [y_mid, y_mid],
                          color=orig_handle.linecolor,
                          linestyle='-',
                          linewidth=1)
        handlebox.add_artist(line)
        return line


class UncertObject_single(object):
    def __init__(self, color):
        self.color = color


class uncert_handler_single(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        x0 = x0 + 0.5 * width
        rec = mpatches.Rectangle([x0 + width, y0], 0.5 * width,
                                 height,
                                 facecolor=orig_handle.color,
                                 edgecolor=orig_handle.color,
                                 transform=handlebox.get_transform(),
                                 zorder=10)
        handlebox.add_artist(rec)
        return rec

handler_mapper = {DataObject: data_handler(),
                  UncertObject: uncert_handler(),
                  UncertObject_single: uncert_handler_single()}


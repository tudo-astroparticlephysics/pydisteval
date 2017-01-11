import logging

import numpy as np

from .base_classes import CalcPart, PlotPart
from .functions import plot_funcs

class CalcBinning(CalcPart):
    name = 'CalcBinning'
    level = 0
    def __init__(self, n_bins=50, check_all=True):
        super(CalcBinning, self).__init__()
        self.n_bins = n_bins
        self.check_all = check_all

    def execute(self, result_tray, component):
        super(CalcBinning, self).execute(result_tray, component)
        if not hasattr(result_tray, 'binning'):
            min_x = np.min(component.X)
            max_x = np.max(component.X)
            binning = np.linspace(min_x, max_x, self.n_bins + 1)
            result_tray.add(binning, 'binning')
        elif self.check_all:
            current_min_x = result_tray.binning[0]
            current_max_x = result_tray.binning[-1]
            min_x = min(current_min_x, np.min(component.X))
            max_x = max(current_max_x, np.max(component.X))
            binning = np.linspace(min_x, max_x, self.n_bins + 1)
            result_tray.add(binning, 'binning')
        return result_tray


class CalcHistogram(CalcPart):
    name = 'CalcHistogram'
    level = 1

    def execute(self, result_tray, component):
        super(CalcHistogram, self).execute(result_tray, component)
        if not hasattr(result_tray, 'binning'):
            raise RuntimeError('No \'binning\' in the result tray.'
                               ' run \'CalcBinning\' first!')
        else:
            binning = result_tray.binning

        weights = component.weights
        X = component.X
        idx = component.idx

        n_bins = len(binning) + 1
        if component.c_type == 'ref':
            result_tray.add(component.livetime, 'ref_livetime')
        if not hasattr(result_tray, 'sum_w'):
            sum_w = np.zeros((n_bins, result_tray.n_components))
            sum_w_squared = np.zeros_like(sum_w)
        else:
            sum_w = result_tray.sum_w
            sum_w_squared = result_tray.sum_w_squared

        digitized = np.digitize(X, bins=binning)
        sum_w[:, idx] = np.bincount(digitized,
                                    weights=weights,
                                    minlength=n_bins)
        if weights is not None:
            sum_w_squared[:, idx] = np.bincount(digitized,
                                                weights=weights**2,
                                                minlength=n_bins)
        else:
            sum_w_squared[:, idx] = sum_w[:, idx]
        result_tray.add(sum_w, 'sum_w')
        result_tray.add(sum_w_squared, 'sum_w_squared')
        return result_tray


class CalcAggarwalHistoErrors(CalcPart):
    name = 'CalcAggarwalHistoErrors'
    def __init__(self, alphas):
        super(CalcAggarwalHistoErrors, self).__init__()
        self.alphas = alphas
        self.ref_idx = None

    def execute(self, result_tray, component):
        super(CalcAggarwalHistoErrors, self).execute(result_tray, component)
        raise NotImplementedError

    def finish(self, result_tray):
        super(CalcAggarwalHistoErrors, self).finish(self, result_tray)
        self.ref_idx = None


class CalcClassicHistoErrors(CalcPart):
    name = 'CalcClassicHistoErrors'

    def execute(self, result_tray, component):
        super(CalcClassicHistoErrors, self).execute(result_tray, component)

        if not hasattr(result_tray, 'sum_w'):
            raise RuntimeError('No \'sum_w\' in the result tray.'
                               ' run \'CalcHistogram\' first!')
        else:
            sum_w = result_tray.sum_w
            sum_w_squared = result_tray.sum_w_squared

        if not hasattr(result_tray, 'rel_std'):
            rel_std = np.zeros_like(sum_w)
        else:
            rel_std = result_tray.rel_std
        idx = component.idx
        abs_std = np.sqrt(sum_w_squared[:, idx])
        mask = abs_std > 0
        rel_std[mask, component.idx] = abs_std[mask] / sum_w[mask, idx]
        result_tray.add(rel_std, 'rel_std')
        return result_tray


class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    rows = 5
    def __init__(self):
        super(PlotHistAggerwal, self).__init__()

    def execute(self, result_tray, component):
        super(PlotHistAggerwal, self).execute(result_tray, component)
        self.ax


class PlotHistClassic(PlotPart):
    name = 'PlotHistClassic'
    rows = 5
    def __init__(self,
                 log_y,
                 normalize,
                 bands,
                 band_borders,
                 band_brighten,
                 band_alpha):
        super(PlotHistClassic, self).__init__()
        self.log_y = log_y
        self.bands = bands
        self.normalize = normalize
        self.band_borders = band_borders
        self.band_brighten = band_brighten
        self.band_alpha = band_alpha
        self.y_lower = None

    def first_execute(self, result_tray, component):
        super(PlotHistClassic, self).first_execute(result_tray, component)
        if self.log_y:
            self.ax.set_yscale('log', clip=True)
        self.ax.set_ylabel('Frequence')
        return result_tray

    def execute(self, result_tray, component):
        super(PlotHistClassic, self).execute(result_tray, component)
        idx = component.idx
        label = component.label
        color = component.color
        binning = result_tray.binning
        bin_mids = (binning[1:] + binning[:-1]) / 2.

        y_vals = result_tray.sum_w[1:-1, idx]
        y_std = result_tray.rel_std[:, idx][1:-1] * y_vals
        y_low = y_vals - y_std
        y_high = y_vals + y_std

        if self.bands:
            plot_funcs.plot_hist(ax=self.ax,
                                 bin_edges=binning,
                                 y=y_vals,
                                 color=color)
            plot_funcs.plot_band(ax=self.ax,
                                 bin_edges=binning,
                                 y_err_low=y_low,
                                 y_err_high=y_high,
                                 color=color,
                                 alpha=self.band_alpha,
                                 borders=self.band_borders,
                                 brighten=self.band_brighten)
        else:
            plot_funcs.plot_hist(ax=self.ax,
                                 bin_edges=binning,
                                 y=y_vals,
                                 color=color,
                                 yerr=y_std)
        if self.log_y:
            y_min = np.min(y_vals[y_vals > 0])
            if self.y_lower is None:
                self.y_lower = y_min
            else:
                self.y_lower = min(self.y_lower, y_min)
        return result_tray

    def finish(self, result_tray):
        super(PlotHistClassic, self).finish(result_tray)
        current_y_lims = self.ax.get_ylim()
        self.ax.set_ylim([self.y_lower * 0.5, current_y_lims[1]])

class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    rows = 1
    def __init__(self):
        super(PlotRatioAggerwal, self).__init__()

    def execute(self, result_tray, component):
        super(PlotRatioAggerwal, self).execute(result_tray, component)
        raise NotImplementedError


class PlotRatioClassic(PlotPart):
    name = 'PlotRatioClassic'
    rows = 1.5
    def __init__(self,
                 bands,
                 band_borders,
                 band_brighten,
                 band_alpha,
                 y_label=r'$\frac{\mathregular{Test - Ref}}{\sigma}$',
                 y_lims=None):
        super(PlotRatioClassic, self).__init__()
        self.bands = bands
        self.band_borders = band_borders
        self.band_brighten = band_brighten
        self.band_alpha = band_alpha
        self.y_label = y_label
        self.y_lims = y_lims
        self.abs_max = None

    def first_execute(self, result_tray, component):
        super(PlotRatioClassic, self).first_execute(result_tray, component)
        self.ax.set_ylabel(self.y_label)
        return result_tray

    def execute(self, result_tray, component):
        super(PlotRatioClassic, self).execute(result_tray, component)
        if component.c_type == 'ref':
            self.__execute_ref__(result_tray, component)
        else:
            self.__execute_others__(result_tray, component)
        return result_tray

    def __execute_ref__(self, result_tray, component):
        binning = result_tray.binning
        color = component.color

        plot_funcs.plot_hist(ax=self.ax,
                             bin_edges=binning,
                             y=np.zeros(len(binning) - 1),
                             color=color,
                             yerr=None)
        return result_tray

    def __execute_others__(self, result_tray, component):
        ref_idx = result_tray.ref_idx
        idx = component.idx
        label = component.label
        color = component.color
        binning = result_tray.binning
        bin_mids = (binning[1:] + binning[:-1]) / 2.

        y_vals = result_tray.sum_w[1:-1, idx]
        y_std = result_tray.rel_std[:, idx][1:-1] * y_vals
        ref_vals = result_tray.sum_w[1:-1, ref_idx]
        ref_std = result_tray.rel_std[:, ref_idx][1:-1] * ref_vals

        ratio = np.empty_like(y_vals)
        ratio[:] = np.nan
        mask = y_std > 0
        total_std = np.empty_like(ratio)
        total_std[:] = np.nan
        total_std[mask] = np.sqrt(y_std[mask]**2 + ref_std[mask]**2)

        ratio[mask] = (ref_vals[mask] - y_vals[mask]) / total_std[mask]

        if self.bands:
            plot_funcs.plot_hist(ax=self.ax,
                                 bin_edges=binning,
                                 y=ratio,
                                 color=color)
            plot_funcs.plot_band(ax=self.ax,
                                 bin_edges=binning,
                                 y_err_low=ratio - np.ones_like(ratio),
                                 y_err_high=ratio + np.ones_like(ratio),
                                 color=color,
                                 alpha=self.band_alpha,
                                 borders=self.band_borders,
                                 brighten=self.band_brighten)
        else:
            plot_funcs.plot_hist(ax=self.ax,
                                 bin_edges=binning,
                                 y=ratio,
                                 color=color,
                                 yerr=np.ones_like(ratio))

        abs_max = np.max(np.absolute(ratio[mask]))
        if self.y_lims is None:
            if self.abs_max is None:
                self.abs_max = abs_max
            else:
                self.abs_max = max(self.abs_max, abs_max)

        return result_tray

    def finish(self, result_tray):
        super(PlotRatioClassic, self).finish(result_tray)
        current_y_lims = self.ax.get_ylim()
        if self.y_lims is None:
            self.ax.set_ylim([self.abs_max * -1.5,
                              self.abs_max * 1.5])
        else:
            self.ax.set_ylim(self.y_lims)

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from .base_classes import CalcPart, PlotPart
from .functions import plot_funcs
from .functions import calc_funcs
from .functions import legend_entries as le


class CalcBinning(CalcPart):
    name = 'CalcBinning'
    level = 0

    def __init__(self, n_bins=50, check_all=True):
        super(CalcBinning, self).__init__()
        self.n_bins = n_bins
        self.check_all = check_all

    def execute(self, result_tray, component):
        result_tray = super(CalcBinning, self).execute(result_tray, component)
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
        result_tray = super(CalcHistogram, self).execute(result_tray,
                                                         component)
        if not hasattr(result_tray, 'binning'):
            raise RuntimeError('No \'binning\' in the result tray.'
                               ' run \'CalcBinning\' first!')
        else:
            binning = result_tray.binning

        weights = component.weights
        X = component.X
        idx = component.idx

        n_bins = len(binning) + 1
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

    def __init__(self, alpha):
        super(CalcAggarwalHistoErrors, self).__init__()
        self.alpha = alpha

    def start(self, result_tray):
        result_tray = super(CalcAggarwalHistoErrors, self).start(result_tray)
        result_tray.add(self.alpha, 'alpha')
        return result_tray

    def execute(self, result_tray, component):
        result_tray = super(CalcAggarwalHistoErrors, self).execute(result_tray,
                                                                   component)
        if component.c_type == 'ref':
            if not hasattr(result_tray, 'sum_w'):
                raise RuntimeError('No \'sum_w\' in the result tray.'
                                   ' run \'CalcHistogram\' first!')
            else:
                sum_w = result_tray.sum_w
                sum_w_squared = result_tray.sum_w_squared

            scale_factor = result_tray.test_livetime / component.livetime
            mu = sum_w[:, component.idx] * scale_factor

            rel_std = np.empty((len(sum_w), len(self.alpha), 2))
            rel_std[:] = np.nan

            lower, upper = calc_funcs.aggarwall_limits(mu,
                                                       alpha=self.alpha)
            mask = mu > 0
            for i in range(len(self.alpha)):
                rel_std[mask, i, 0] = lower[mask, i] / mu[mask]
                rel_std[mask, i, 1] = upper[mask, i] / mu[mask]
            result_tray.add(rel_std, 'rel_std_aggarwal')
        return result_tray


class CalcClassicHistoErrors(CalcPart):
    name = 'CalcClassicHistoErrors'

    def execute(self, result_tray, component):
        result_tray = super(CalcClassicHistoErrors, self).execute(result_tray,
                                                                  component)
        if not hasattr(result_tray, 'sum_w'):
            raise RuntimeError('No \'sum_w\' in the result tray.'
                               ' run \'CalcHistogram\' first!')
        else:
            sum_w = result_tray.sum_w
            sum_w_squared = result_tray.sum_w_squared

        if not hasattr(result_tray, 'rel_std_classic'):
            rel_std = np.zeros_like(sum_w)
        else:
            rel_std = result_tray.rel_std_classic
        idx = component.idx
        abs_std = np.sqrt(sum_w_squared[:, idx])
        mask = abs_std > 0
        rel_std[mask, component.idx] = abs_std[mask] / sum_w[mask, idx]
        result_tray.add(rel_std, 'rel_std_classic')
        return result_tray


class CalcNormalization(CalcPart):
    name = 'CalcNormalization'
    level = 2

    def __init__(self,
                 normalize):
        super(CalcNormalization, self).__init__()
        self.normalize = normalize

    def execute(self, result_tray, component):
        result_tray = super(CalcNormalization, self).execute(result_tray,
                                                             component)
        sum_w = result_tray.sum_w
        if self.normalize == 'test_livetime':
            scaling = result_tray.test_livetime / component.livetime
        elif self.normalize == 'livetime':
            scaling = 1. / component.livetime
        elif self.normalize == 'sum_w':
            scaling = 1. / np.sum(sum_w)
        else:
            scaling = 1.
        sum_w[:, component.idx] *= scaling
        result_tray.add(sum_w, 'sum_w')
        return result_tray


class PlotHistClassic(PlotPart):
    name = 'PlotHistClassic'
    rows = 5

    def __init__(self,
                 log_y,
                 bands,
                 band_borders,
                 band_brighten,
                 band_alpha):
        super(PlotHistClassic, self).__init__()
        self.log_y = log_y
        self.bands = bands
        self.band_borders = band_borders
        self.band_brighten = band_brighten
        self.band_alpha = band_alpha
        self.leg_labels = []
        self.leg_entries = []
        self.y_lower = None

    def start(self, result_tray):
        result_tray = super(PlotHistClassic, self).start(result_tray)
        if self.log_y:
            self.ax.set_yscale('log', clip=True)
        self.ax.set_ylabel('Frequence')
        return result_tray

    def execute(self, result_tray, component):
        result_tray = super(PlotHistClassic, self).execute(result_tray,
                                                           component)
        idx = component.idx
        label = component.label
        color = component.color
        binning = result_tray.binning
        bin_mids = (binning[1:] + binning[:-1]) / 2.
        y_vals = result_tray.sum_w[1:-1, component.idx]
        y_std = result_tray.rel_std_classic[1:-1, idx] * y_vals
        y_low = y_vals - y_std
        y_high = y_vals + y_std

        if self.bands:
            leg_obj = plot_funcs.plot_hist(ax=self.ax,
                                           bin_edges=binning,
                                           y=y_vals,
                                           color=color)
            _ = plot_funcs.plot_band(ax=self.ax,
                                     bin_edges=binning,
                                     y_err_low=y_low,
                                     y_err_high=y_high,
                                     color=color,
                                     alpha=1.0,
                                     borders=False,
                                     brighten=False)
        else:
            leg_obj = plot_funcs.plot_hist(ax=self.ax,
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
        self.leg_entries.append(leg_obj)
        self.leg_labels.append(component.label)
        return result_tray

    def finish(self, result_tray):
        result_tray = super(PlotHistClassic, self).finish(result_tray)
        if self.log_y:
            current_y_lims = self.ax.get_ylim()
            self.ax.set_ylim([self.y_lower * 0.5, current_y_lims[1]])
        self.ax.legend(self.leg_entries,
                       self.leg_labels,
                       loc='best',
                       prop={'size': 11})


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

    def start(self, result_tray):
        result_tray = super(PlotRatioClassic, self).start(result_tray)
        self.ax.set_ylabel(self.y_label)
        return result_tray

    def execute(self, result_tray, component):
        super_func = super(PlotRatioClassic, self).execute
        result_tray = super_func(result_tray, component)
        if component.c_type == 'ref':
            self.__execute_ref__(result_tray, component)
        elif component.c_type == 'test':
            self.__execute_test__(result_tray, component)
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

    def __execute_test__(self, result_tray, component):
        ref_idx = result_tray.ref_idx
        idx = component.idx
        color = component.color
        binning = result_tray.binning

        y_vals = result_tray.sum_w[1:-1, idx]
        y_std = result_tray.rel_std_classic[:, idx][1:-1] * y_vals
        ref_vals = result_tray.sum_w[1:-1, ref_idx]
        ref_std = result_tray.rel_std_classic[:, ref_idx][1:-1] * ref_vals

        ratio = np.empty_like(y_vals)
        ratio[:] = np.nan
        mask = y_std > 0
        total_std = np.empty_like(ratio)
        total_std[:] = np.nan
        total_std[mask] = np.sqrt(y_std[mask]**2 + ref_std[mask]**2)

        ratio[mask] = (ref_vals[mask] - y_vals[mask]) / total_std[mask]

        if self.bands:
            plot_funcs.plot_line(ax=self.ax,
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
        result_tray = super(PlotRatioClassic, self).finish(result_tray)
        if self.y_lims is None:
            self.ax.set_ylim([self.abs_max * -1.5,
                              self.abs_max * 1.5])
        else:
            self.ax.set_ylim(self.y_lims)


class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    rows = 5

    def __init__(self,
                 log_y,
                 bands,
                 band_borders,
                 band_brighten,
                 band_alpha):
        super(PlotHistAggerwal, self).__init__()
        self.log_y = log_y
        self.bands = bands
        self.y_lower = None
        self.leg_labels = []
        self.leg_entries = []

    def start(self, result_tray):
        result_tray = super(PlotHistAggerwal, self).start(result_tray)
        if self.log_y:
            self.ax.set_yscale('log', clip=True)
        self.ax.set_ylabel('Frequence')
        return result_tray

    def execute(self, result_tray, component):
        result_tray = super(PlotHistAggerwal, self).execute(result_tray,
                                                            component)
        y_vals = result_tray.sum_w[1:-1, component.idx]
        if self.log_y:
            y_min = np.min(y_vals[y_vals > 0])
            if self.y_lower is None:
                self.y_lower = y_min
            else:
                self.y_lower = min(self.y_lower, y_min)

        if component.c_type in ['ref', 'ref_part']:
            if component.c_type == 'ref':
                part = False
            else:
                part = True
            leg_objs, labels = self.__execute_ref__(result_tray,
                                                    component,
                                                    part=part)
        elif component.c_type in ['test', 'test_part']:
            if component.c_type == 'test':
                part = False
            else:
                part = True
            leg_objs, labels = self.__execute_test__(result_tray,
                                                     component,
                                                     part=part)

        self.leg_labels.extend(labels)
        self.leg_entries.extend(leg_objs)
        return result_tray

    def __execute_test__(self, result_tray, component, part=False):
        y_vals = result_tray.sum_w[1:-1, component.idx]
        if part:
            leg_obj = plot_funcs.plot_data_style(fig=result_tray.fig,
                                                 ax=self.ax,
                                                 bin_edges=result_tray.binning,
                                                 y=y_vals,
                                                 facecolor='none',
                                                 edgecolor=component.color,
                                                 alpha=1.0)
        else:
            leg_obj = plot_funcs.plot_data_style(fig=result_tray.fig,
                                                 ax=self.ax,
                                                 bin_edges=result_tray.binning,
                                                 y=y_vals,
                                                 facecolor=component.color,
                                                 edgecolor='k',
                                                 alpha=1.0)
        return [leg_obj], [component.label]

    def __execute_ref__(self, result_tray, component, part=False):
        y_vals = result_tray.sum_w[1:-1, component.idx]
        line_obj = plot_funcs.plot_line(ax=self.ax,
                                        bin_edges=result_tray.binning,
                                        y=y_vals,
                                        color=component.color)
        if part:
            labels = [component.label]
            leg_objs = [line_obj]
        else:
            leg_objs = plot_funcs.plot_uncertainties(
                ax=self.ax,
                bin_edges=result_tray.binning,
                y=y_vals,
                uncert=result_tray.rel_std_aggarwal[1:-1],
                color=component.color,
                cmap=component.cmap,
                alpha=result_tray.alpha)
            labels = [component.label]
            for a_i in result_tray.alpha:
                labels.append('      %.1f%% Uncert.' % (a_i * 100.))
        return leg_objs, labels

    def finish(self, result_tray):
        result_tray = super(PlotHistAggerwal, self).finish(result_tray)
        if self.log_y:
            current_y_lims = self.ax.get_ylim()
            self.ax.set_ylim([self.y_lower * 0.5, current_y_lims[1]])
        self.ax.legend(self.leg_entries,
                       self.leg_labels,
                       handler_map=le.handler_mapper,
                       loc='best',
                       prop={'size': 11})


class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    rows = 2

    def __init__(self, zoomed):
        super(PlotRatioAggerwal, self).__init__()
        self.zoomed = zoomed
        if zoomed:
            self.rows = 3

    def set_ax(self, fig, total_parts, idx, x0, x1, y0, y1):
        self.logger.debug('\t{}: Setting up Axes!'.format(self.name))
        if idx == 0:
            self.is_top = True
            top_offset = self.large_offset
        else:
            self.is_bot = False
            top_offset = self.small_offset
        if idx == total_parts - 1:
            self.is_bot = True
            bot_offset = self.large_offset
        else:
            self.is_bot = False
            bot_offset = self.small_offset
        if self.zoomed:
            self.gs = GridSpec(2, 1,
                               left=x0 + 0.1,
                               right=x1 - 0.1,
                               top=y1 - top_offset,
                               bottom=y0 + bot_offset,
                               hspace=0.0)
            self.ax_upper = plt.subplot(self.gs[0, :])
            self.ax = plt.subplot(self.gs[1, :])
            plt.setp(self.ax_upper.get_xticklabels(), visible=False)
        else:
            self.gs = GridSpec(1, 1,
                               left=x0 + 0.1,
                               right=x1 - 0.1,
                               top=y1 - top_offset,
                               bottom=y0 + bot_offset)
            self.ax = plt.subplot(self.gs[:, :])
        return self.ax

    def get_ax(self):
        return [self.ax, self.ax_upper]

    def finish(self, result_tray):
        super(PlotRatioAggerwal, self).finish(result_tray)
        self.ax_upper.set_xlim([result_tray.binning[0],
                                result_tray.binning[-1]])

    def execute(self, result_tray, component):
        result_tray = super(PlotRatioAggerwal, self).execute(result_tray,
                                                             component)
        return result_tray

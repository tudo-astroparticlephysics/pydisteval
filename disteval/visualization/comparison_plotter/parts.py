from __future__ import absolute_import, print_function, division
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from .base_classes import CalcPart, PlotPart
from .functions import plot_funcs
from .functions import calc_funcs
from .functions import likelihoods
from .functions import legend_entries as le

from tqdm import tqdm


class CalcBinning(CalcPart):
    name = 'CalcBinning'
    level = 0

    def __init__(self, n_bins=50, binning_dict=None, check_all=False):
        super(CalcBinning, self).__init__()
        self.n_bins = n_bins
        self.check_all = check_all
        self.binning_dict = binning_dict

    def execute(self, result_tray, component):
        result_tray = super(CalcBinning, self).execute(result_tray, component)
        if self.binning_dict is not None:
            if not isinstance(self.binning_dict, dict):
                raise TypeError('\'binning_dict\' must be of type dict!')
            if result_tray.x_label in self.binning_dict.keys():
                binning = self.binning_dict[result_tray.x_label]
                result_tray.add(binning, 'binning')
                return result_tray
        if not hasattr(result_tray, 'binning'):
            min_x = np.min(component.X)
            max_x = np.max(component.X)
            upper_x = max_x + (max_x - min_x) / self.n_bins
            binning = np.linspace(min_x, upper_x, self.n_bins + 2)
            result_tray.add(binning, 'binning')
        elif self.check_all:
            current_min_x = result_tray.binning[0]
            current_max_x = result_tray.binning[-1]
            min_x = min(current_min_x, np.min(component.X))
            max_x = max(current_max_x, np.max(component.X))
            upper_x = max_x + (max_x - min_x) / self.n_bins
            binning = np.linspace(min_x, upper_x, self.n_bins + 2)
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
            k_mc = np.zeros((n_bins, result_tray.n_components))
        else:
            sum_w = result_tray.sum_w
            sum_w_squared = result_tray.sum_w_squared
            k_mc = result_tray.k_mc

        digitized = np.digitize(X, bins=binning)
        sum_w[:, idx] = np.bincount(digitized,
                                    weights=weights,
                                    minlength=n_bins)
        k_mc[:, idx] = np.bincount(digitized,
                                   minlength=n_bins)
        if weights is not None:
            w_list = []
            for bin_i in range(len(binning)+1):
                w_list.append(weights[digitized == bin_i])
            sum_w_squared[:, idx] = np.bincount(digitized,
                                                weights=weights**2,
                                                minlength=n_bins)
        else:
            sum_w_squared[:, idx] = sum_w[:, idx]
            w_list = []
        result_tray.add(sum_w, 'sum_w')
        result_tray.add(sum_w_squared, 'sum_w_squared')
        result_tray.add(k_mc, 'k_mc')
        result_tray.add(w_list, 'w_list')
        return result_tray


class CalcLimitedMCHistoErrors(CalcPart):
    name = 'CalcLimitedMCHistoErrors'

    def __init__(self, alpha, likelihood):
        super(CalcLimitedMCHistoErrors, self).__init__()
        self.alpha = alpha
        self.likelihood = likelihood.lower()

    def start(self, result_tray):
        result_tray = super(CalcLimitedMCHistoErrors, self).start(result_tray)
        result_tray.add(self.alpha, 'alpha')
        return result_tray

    def execute(self, result_tray, component):
        result_tray = super(CalcLimitedMCHistoErrors, self).execute(
            result_tray, component)
        if component.c_type == 'ref':
            scale_factor = result_tray.test_livetime / component.livetime

            if not (hasattr(result_tray, 'sum_w') and
                    hasattr(result_tray, 'k_mc')):
                raise RuntimeError('No \'sum_w\' in the result tray.'
                                   ' run \'CalcHistogram\' first!')
            else:
                sum_w = result_tray.sum_w[:, component.idx] * scale_factor
                sum_w2 = result_tray.sum_w_squared[:, component.idx] * \
                    scale_factor**2
                k_mc = result_tray.k_mc[:, component.idx]
                w_list = result_tray.w_list

            mus = sum_w

            pdfs = []
            ks = []

            for i, mu in tqdm(enumerate(mus)):
                if self.likelihood == 'say':
                    llh_func = likelihoods.SAY_likelihood
                    first_guess = mu
                    llh_kwargs = {
                        'mu': mu,
                        'w2_sum': sum_w2[i]
                    }

                elif self.likelihood == 'thorsten_general':
                    llh_func = likelihoods.poisson_general_weights
                    first_guess = mu
                    llh_kwargs = {
                        'weights': w_list[i] * scale_factor
                    }

                elif self.likelihood == 'thorsten_equal':
                    avg_w = sum_w / k_mc
                    llh_func = likelihoods.poisson_equal_weights
                    first_guess = mu
                    llh_kwargs = {
                        'k_mc': k_mc,
                        'avgweights': avg_w
                    }

                else:
                    raise NotImplementedError(
                        'Chosen likelihood is not implemented. ' +
                        'Choose either \'say\', \'thorsten_general\'' +
                        'or \'thorsten_equal\'!')

                k_range, pdf = calc_funcs.evaluate_normalized_likelihood(
                    llh_func=llh_func,
                    coverage=0.99999999,
                    first_guess=first_guess,
                    **llh_kwargs)

                ks.append(k_range)

                # Example on how to use DimaLlh, which can't be treated the
                # same way because it is unnormalized
                # lower = np.maximum(0, mu - 200)
                # upper = mu + 200
                # k_range = np.arange(int(lower), int(upper))
                # ks.append(k_range)
                # pdf = []

                # for k in k_range:
                #     pdf.append(np.exp(
                #         likelihoods.poisson_general_weights_chirkin_13(
                #             np.array([k]),
                #             w_list[i] * scale_factor,
                #             [np.ones(len(w_list[i])) == 1])))

                # pdf = np.array(pdf) / np.sum(pdf)

                pdfs.append(pdf)

            rel_std = np.empty((len(mus), len(self.alpha), 2))
            rel_std[:] = np.nan

            lower, upper = calc_funcs.aggarwal_limits_pdf(pdfs,
                                                          ks,
                                                          alpha=self.alpha)

            mask = mus > 0
            for i in range(len(self.alpha)):
                rel_std[mask, i, 0] = lower[mask, i] / mus[mask]
                rel_std[mask, i, 1] = upper[mask, i] / mus[mask]
            result_tray.add(rel_std, 'rel_std_aggarwal')
            result_tray.add(pdfs, 'pdfs')
            result_tray.add(ks, 'ks')
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

            scale_factor = result_tray.test_livetime / component.livetime
            mu = sum_w[:, component.idx] * scale_factor

            rel_std = np.empty((len(sum_w), len(self.alpha), 2))
            rel_std[:] = np.nan

            lower, upper = calc_funcs.aggarwal_limits(mu,
                                                      alpha=self.alpha)
            mask = mu > 0
            for i in range(len(self.alpha)):
                rel_std[mask, i, 0] = lower[mask, i] / mu[mask]
                rel_std[mask, i, 1] = upper[mask, i] / mu[mask]
            result_tray.add(rel_std, 'rel_std_aggarwal')
        return result_tray


class CalcLimitedMCRatios(CalcPart):
    name = 'CalcLimitedMCRatios'
    level = 3

    def execute(self, result_tray, component):
        result_tray = super(CalcLimitedMCRatios, self).execute(
            result_tray, component)
        sum_w = result_tray.sum_w
        mu = sum_w[:, result_tray.ref_idx]
        pdfs = result_tray.pdfs
        ks = result_tray.ks
        if component.c_type == 'ref':
            if not hasattr(result_tray, 'rel_std_aggarwal'):
                raise RuntimeError('No \'rel_std_aggarwal\' in the result tray'
                                   '. run \'CalcAggarwalHistoErrors\' first!')
            else:
                rel_std = result_tray.rel_std_aggarwal
            y_mins_limit = [0, 0]
            limits = np.zeros_like(rel_std)
            limits = calc_funcs.calc_p_alpha_limits_pdf(
                pdfs=pdfs,
                ks=ks,
                mu=mu,
                rel_std=rel_std)
            limits_mapped = np.zeros_like(limits)

            limits_mapped[:, :, 0], y_min = calc_funcs.map_aggarwal_limits(
                limits[:, :, 0],
                y_0=1.,
                upper=False)
            y_mins_limit[0] = y_min

            limits_mapped[:, :, 1], y_min = calc_funcs.map_aggarwal_limits(
                limits[:, :, 1],
                y_0=1.,
                upper=True)
            y_mins_limit[1] = y_min

            result_tray.add(y_mins_limit, 'y_mins_limit')
            result_tray.add(limits_mapped, 'limits_mapped')
        if component.c_type == 'test':
            k = sum_w[:, component.idx]
            ratio = calc_funcs.calc_p_alpha_ratio_pdf(pdfs, ks, mu, k)
            upper = k > mu
            below = ~upper
            ratio_mapped = np.array(ratio)
            ratio_mapped[below], y_below = calc_funcs.map_aggarwal_ratio(
                ratio[below],
                y_0=1.,
                upper=False)
            ratio_mapped[upper], y_upper = calc_funcs.map_aggarwal_ratio(
                ratio[upper],
                y_0=1.,
                upper=True)

            result_tray.add([y_below, y_upper], 'y_mins_ratio')
            result_tray.add(ratio_mapped, 'ratio_mapped')
            result_tray.add(upper, 'is_above')
        return result_tray


class CalcAggarwalRatios(CalcPart):
    name = 'CalcAggarwalRatios'
    level = 3

    def execute(self, result_tray, component):
        result_tray = super(CalcAggarwalRatios, self).execute(result_tray,
                                                              component)
        sum_w = result_tray.sum_w
        mu = sum_w[:, result_tray.ref_idx]
        if component.c_type == 'ref':
            if not hasattr(result_tray, 'rel_std_aggarwal'):
                raise RuntimeError('No \'rel_std_aggarwal\' in the result tray'
                                   '. run \'CalcAggarwalHistoErrors\' first!')
            else:
                rel_std = result_tray.rel_std_aggarwal
            y_mins_limit = [0, 0]
            limits = np.zeros_like(rel_std)
            limits = calc_funcs.calc_p_alpha_limits(
                mu=mu,
                rel_std=rel_std)
            limits_mapped = np.zeros_like(limits)

            limits_mapped[:, :, 0], y_min = calc_funcs.map_aggarwal_limits(
                limits[:, :, 0],
                y_0=1.,
                upper=False)
            y_mins_limit[0] = y_min

            limits_mapped[:, :, 1], y_min = calc_funcs.map_aggarwal_limits(
                limits[:, :, 1],
                y_0=1.,
                upper=True)
            y_mins_limit[1] = y_min

            result_tray.add(y_mins_limit, 'y_mins_limit')
            result_tray.add(limits_mapped, 'limits_mapped')
        if component.c_type == 'test':
            k = sum_w[:, component.idx]
            ratio = calc_funcs.calc_p_alpha_ratio(mu, k)
            upper = k > mu
            below = ~upper
            ratio_mapped = np.array(ratio)
            ratio_mapped[below], y_below = calc_funcs.map_aggarwal_ratio(
                ratio[below],
                y_0=1.,
                upper=False)
            ratio_mapped[upper], y_upper = calc_funcs.map_aggarwal_ratio(
                ratio[upper],
                y_0=1.,
                upper=True)

            result_tray.add([y_below, y_upper], 'y_mins_ratio')
            result_tray.add(ratio_mapped, 'ratio_mapped')
            result_tray.add(upper, 'is_above')
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
                 y_label):
        super(PlotHistClassic, self).__init__()
        self.log_y = log_y
        self.leg_labels = []
        self.leg_entries = []
        self.y_lower = None
        self.y_label = y_label

    def start(self, result_tray):
        result_tray = super(PlotHistClassic, self).start(result_tray)
        if self.log_y:
            self.ax.set_yscale('log', nonposy='clip')
        self.ax.set_ylabel(self.y_label)
        return result_tray

    def execute(self, result_tray, component):
        result_tray = super(PlotHistClassic, self).execute(result_tray,
                                                           component)
        idx = component.idx
        color = component.color
        binning = result_tray.binning
        y_vals = result_tray.sum_w[1:-1, component.idx]
        y_std = result_tray.rel_std_classic[1:-1, idx] * y_vals
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
        self.leg_labels = []


class PlotRatioClassic(PlotPart):
    name = 'PlotRatioClassic'
    rows = 1.5

    def __init__(self,
                 y_label=r'$\frac{\mathregular{Test - Ref}}{\sigma}$',
                 y_lims=None):
        super(PlotRatioClassic, self).__init__()
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

        ratio[mask] = (y_vals[mask] - ref_vals[mask]) / total_std[mask]

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
        self.leg_labels = []
        self.leg_entries = []


class PlotHistAggerwal(PlotPart):
    name = 'PlotHistAggerwal'
    rows = 5

    def __init__(self,
                 log_y,
                 bands,
                 band_borders,
                 band_brighten,
                 band_alpha,
                 y_label):
        super(PlotHistAggerwal, self).__init__()
        self.log_y = log_y
        self.bands = bands
        self.y_lower = None
        self.leg_labels = []
        self.leg_entries = []
        self.y_label = y_label

    def start(self, result_tray):
        result_tray = super(PlotHistAggerwal, self).start(result_tray)
        if self.log_y:
            self.ax.set_yscale('log', nonposy='clip')
        self.ax.set_ylabel(self.y_label)
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
            rel_std = result_tray.rel_std_aggarwal[1:-1]
            abs_std = np.zeros_like(rel_std)
            for i in range(abs_std.shape[1]):
                for j in range(abs_std.shape[2]):
                    abs_std[:, i, j] = rel_std[:, i, j] * y_vals

            leg_objs = plot_funcs.plot_uncertainties(
                ax=self.ax,
                bin_edges=result_tray.binning,
                uncert=abs_std,
                color=component.color,
                cmap=component.cmap)
            labels = [component.label]
            for a_i in result_tray.alpha:
                labels.append('{:.1f}\% Interval'.format(a_i * 100.))
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
        self.leg_labels = []
        self.leg_entries = []


class PlotRatioAggerwal(PlotPart):
    name = 'PlotRatioAggerwal'
    rows = 2
    zoom = -5

    def __init__(self, zoomed, y_label):
        super(PlotRatioAggerwal, self).__init__()
        self.zoomed = zoomed
        self.y_label = y_label
        if zoomed:
            self.rows = 3

    def start(self, result_tray):
        y_min_limit = np.min(result_tray.y_mins_limit)
        y_min_ratio = np.min(result_tray.y_mins_ratio)
        self.y_min = min(y_min_limit, y_min_ratio)

    def set_ax(self, fig, total_parts, idx, x0, x1, y0, y1,
               medium_offsets_only=False):
        self.logger.debug(u'\t{}: Setting up Axes!'.format(self.name))
        self.is_top = False
        self.is_bot = False

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

        if medium_offsets_only:
            if self.is_top:
                top_offset = self.medium_offset
            if self.is_bot:
                bot_offset = self.medium_offset

        if self.zoomed:
            self.gs = GridSpec(2, 1,
                               left=x0 + 0.1,
                               right=x1 - 0.1,
                               top=y1 - top_offset,
                               bottom=y0 + bot_offset,
                               hspace=0.0)
            self.ax_upper = plt.subplot(self.gs[0, :])
            self.ax_upper.set_ylim(-1, 1)
            self.ax = plt.subplot(self.gs[1, :])
            self.ax.set_ylim(-1, 1)
            plt.setp(self.ax_upper.get_xticklabels(), visible=False)
        else:
            self.gs = GridSpec(1, 1,
                               left=x0 + 0.1,
                               right=x1 - 0.1,
                               top=y1 - top_offset,
                               bottom=y0 + bot_offset)
            self.ax = plt.subplot(self.gs[:, :])
            self.ax.set_ylim(-1, 1)
        return self.ax

    def get_ax(self):
        if self.zoomed:
            return [self.ax, self.ax_upper]
        else:
            return self.ax

    def finish(self, result_tray):
        super(PlotRatioAggerwal, self).finish(result_tray)
        if self.zoomed:
            self.ax_upper.set_xlim([result_tray.binning[0],
                                    result_tray.binning[-1]])

    def execute(self, result_tray, component):
        result_tray = super(PlotRatioAggerwal, self).execute(result_tray,
                                                             component)
        if component.c_type == 'test':
            self.__execute_test__(result_tray, component)
        elif component.c_type == 'ref':
            self.__execute_ref__(result_tray, component)
        return result_tray

    def __execute_test__(self, result_tray, component):
        y_mins_ratio = result_tray.y_mins_ratio
        ratio_mapped = result_tray.ratio_mapped[1:-1]
        is_above = result_tray.is_above[1:-1]
        binning = result_tray.binning

        ratio_unzoomed = np.array(ratio_mapped)
        ratio_unzoomed[~is_above] = calc_funcs.rescale_ratio(
            ratio_mapped[~is_above],
            y_mins_ratio[0],
            self.y_min)
        ratio_unzoomed[is_above] = calc_funcs.rescale_ratio(
            ratio_mapped[is_above],
            y_mins_ratio[1],
            self.y_min)

        if not self.zoomed:
            annotation = None
        else:
            annotation = 'Ratio: Smallest Value'

        self.y_min_scaled = self.__plot_test_marker__(
            fig=result_tray.fig,
            ax=self.ax,
            binning=binning,
            ratio=ratio_unzoomed,
            is_above=is_above,
            y_min=self.y_min,
            facecolor=component.color,
            edgecolor='k',
            alpha=1.,
            annotation=annotation)
        if self.zoomed:
            ratio_zoomed = np.array(ratio_mapped)
            ratio_zoomed[~is_above] = calc_funcs.rescale_ratio(
                ratio_mapped[~is_above],
                y_mins_ratio[0],
                self.zoom)
            ratio_zoomed[is_above] = calc_funcs.rescale_ratio(
                ratio_mapped[is_above],
                y_mins_ratio[1],
                self.zoom)
            self.__plot_test_marker__(
                fig=result_tray.fig,
                ax=self.ax_upper,
                binning=binning,
                ratio=ratio_zoomed,
                is_above=is_above,
                y_min=self.zoom,
                facecolor=component.color,
                edgecolor='k',
                alpha=1.,
                annotation='Ratio: Intervals')

    def __execute_ref__(self, result_tray, component):
        binning = result_tray.binning
        y_mins_limit = result_tray.y_mins_limit
        limits_mapped = result_tray.limits_mapped[1:-1]
        limits_unzoomed = np.copy(limits_mapped)
        limits_unzoomed[:, :, 0] = calc_funcs.rescale_limit(
            limits_mapped[:, :, 0],
            y_mins_limit[0],
            self.y_min)
        limits_unzoomed[:, :, 1] = calc_funcs.rescale_limit(
            limits_mapped[:, :, 1],
            y_mins_limit[1],
            self.y_min)
        self.__plot_ref_bands__(ax=self.ax,
                                binning=binning,
                                limits=limits_unzoomed,
                                color=component.color,
                                cmap=component.cmap,
                                alphas=result_tray.alpha)
        if self.zoomed:
            limits_zoomed = np.copy(limits_mapped)
            limits_zoomed[:, :, 0] = calc_funcs.rescale_limit(
                limits_mapped[:, :, 0],
                y_mins_limit[0],
                self.zoom)
            limits_zoomed[:, :, 1] = calc_funcs.rescale_limit(
                limits_mapped[:, :, 1],
                y_mins_limit[1],
                self.zoom)
            self.__plot_ref_bands__(ax=self.ax_upper,
                                    binning=binning,
                                    limits=limits_zoomed,
                                    color=component.color,
                                    cmap=component.cmap,
                                    alphas=result_tray.alpha)

    def __plot_ref_bands__(self,
                           ax,
                           binning,
                           limits,
                           color,
                           cmap,
                           alphas):
        plot_funcs.plot_line(ax=ax,
                             bin_edges=binning,
                             y=np.zeros(len(binning) - 1),
                             color=color)
        plot_funcs.plot_uncertainties(ax=ax,
                                      bin_edges=binning,
                                      uncert=limits,
                                      color=color,
                                      cmap=cmap)

    def __plot_test_marker__(self,
                             fig,
                             ax,
                             binning,
                             ratio,
                             is_above,
                             y_min,
                             facecolor,
                             edgecolor,
                             alpha,
                             annotation=None):
        plot_funcs.plot_test_ratio_mapped(fig=fig,
                                          ax=ax,
                                          bin_edges=binning,
                                          ratio=ratio,
                                          is_above=is_above,
                                          facecolor=facecolor,
                                          edgecolor=edgecolor,
                                          alpha=alpha)
        M_t, M_p, m_t, m_p = plot_funcs.generate_ticks_for_aggarwal_ratio(
            y_0=1.,
            y_min=y_min)
        ax.set_yticklabels(M_t)
        ax.set_yticks(M_p)
        ax.set_yticks(m_p, minor=True)
        ax.set_ylabel(self.y_label)
        ax.yaxis.grid(True)
        if annotation is not None:
            ax.text(binning[1],
                    0.90,
                    annotation,
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=12,
                    color='0.4',
                    alpha=0.7)
        return y_min

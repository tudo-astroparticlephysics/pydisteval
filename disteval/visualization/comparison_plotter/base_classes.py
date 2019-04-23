from __future__ import absolute_import, print_function, division
import logging
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


class Part(object):
    name = 'BasePart'
    level = 1
    logger = logging.getLogger("Part    ")

    def __init__(self):
        self.n_executions = 0

    def execute(self, result_tray, component):
        if self.n_executions == 0:
            result_tray = self.start(result_tray)
        self.logger.debug(u'\t{}: Executing {}!'.format(self.name,
                                                        component.idx))
        self.n_executions += 1
        return result_tray

    def start(self, result_tray):
        self.logger.debug(u'\t{}: Starting!'.format(self.name))
        return result_tray

    def finish(self, result_tray=None):
        self.logger.debug(u'\t{}: Finishing!'.format(self.name))
        self.n_executions = 0
        return result_tray

    def __lt__(self, other):
        return self.level < other.level

    def __eq__(self, other):
        if isinstance(other, Part):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other


class CalcPart(Part):
    pass


class PlotPart(Part):
    rows = 1
    large_offset = 0.08
    small_offset = 0.005
    medium_offset = 0.04

    def init(self):
        self.ax = None

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        return result_tray

    def set_ax(self, fig, total_parts, idx, x0, x1, y0, y1,
               medium_offsets_only=False):
        self.logger.debug(u'\t{}: Setting up Axes!'.format(self.name))
        self.is_bot = False
        self.is_top = False

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

        self.gs = GridSpec(1, 1,
                           left=x0 + 0.1,
                           right=x1 - 0.1,
                           top=y1 - top_offset,
                           bottom=y0 + bot_offset)
        self.ax = plt.subplot(self.gs[:, :])
        return self.ax

    def get_ax(self):
        return self.ax

    def get_rows(self):
        assert self.rows > 0, '\'rows\' must be greater 0!'
        return self.rows

    def finish(self, result_tray):
        super(PlotPart, self).finish(result_tray)
        self.ax.set_xlim([result_tray.binning[0],
                          result_tray.binning[-1]])
        if self.is_bot:
            self.ax.set_xlabel(result_tray.x_label)
        else:
            plt.setp(self.ax.get_xticklabels(), visible=False)
        return result_tray


class Element(object):
    name = 'DefaultElement'
    logger = logging.getLogger("Element ")

    def __init__(self):
        self.plot_components = []
        self.calc_components = []

    def register(self, comparator):
        for calc_comp_i in self.calc_components:
            comparator._register_calc_part(calc_comp_i)
        for plot_comp_i in self.plot_components:
            comparator._register_plot_part(plot_comp_i)

    def __eq__(self, other):
        if isinstance(other, Part):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other


class ResultTray(object):
    def add(self, obj, name):
        setattr(self, name, obj)

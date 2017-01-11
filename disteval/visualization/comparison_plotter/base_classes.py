import logging
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class Part:
    name = 'BasePart'
    level = 1
    logger = logging.getLogger("Part    ")
    def __init__(self):
        self.n_executions = 0

    def execute(self, result_tray, component):
        self.logger.debug('\t{}: Executing {}!'.format(self.name,
                                                       component.idx))
        if self.n_executions == 0:
            result_tray = self.first_execute(result_tray, component)
        self.n_executions += 1
        return result_tray

    def first_execute(self, result_tray, component):
        return result_tray

    def finish(self, result_tray=None):
        self.logger.debug('\t{}: Finishing!'.format(self.name))
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

    def init(self):
        self.ax = None

    def execute(self, result_tray, component):
        super(PlotPart, self).execute(result_tray, component)
        self.first_execute(result_tray, component)

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
        height = y1 - y0
        width = x1 - x0
        self.gs = GridSpec(1, 1,
                           left=x0 + 0.1,
                           right=x1 - 0.1,
                           top=y1 - top_offset,
                           bottom=y0 + bot_offset)
        self.ax = plt.subplot(self.gs[:,:])
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


class Element:
    name = 'DefaultElement'
    plot_components = []
    calc_components = []
    logger = logging.getLogger("Element ")
    def __init__(self):
        pass

    def register(self, comparator):
        for calc_comp_i in Element.calc_components:
            comparator.register_calc_part(calc_comp_i)
        for plot_comp_i in Element.plot_components:
            comparator.register_plot_part(plot_comp_i)

    def __eq__(self, other):
        if isinstance(other, Part):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other




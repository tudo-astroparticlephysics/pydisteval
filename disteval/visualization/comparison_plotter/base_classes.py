import logging


class Part:
    name = 'BasePart'
    level = 1
    logger = logging.getLogger("Part")
    def __init__(self):
        pass

    def execute(self, result_tray, component):
        self.logger.debug('\tExecuting {} for {}!'.format(self.name,
                                                       component.idx))

    def reset(self):
        self.logger.debug('\tResetting {}!'.format(self.name))

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
    def init(self):
        self.ax = None

    def set_ax(self, fig, grid_spec_slice):
        self.logger.debug('Setting up Axes!')
        self.ax = fig.add_subplot(grid_spec_slice)
        return self.ax

    def get_rows(self):
        assert self.rows > 0 and isinstance(self.rows, int), '\'rows\' ' \
            'must be int and greater 0'
        return self.rows


class Element:
    name = 'DefaultElement'
    plot_components = []
    calc_components = []
    logger = logging.getLogger("Element")
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




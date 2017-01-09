from inspect import isclass
import logging

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from . import elements
from .components import Component
from .result_tray import ResultTray

REGISTERED_ELEMENTS = {'aggarwalhisto': elements.AggarwalHisto,
                       'aggarwalratio': elements.AggarwalHisto,
                       'classichisto': elements.ClassicHisto,
                       'classicratio': elements.ClassicRatio}

logger = logging.getLogger("Plotter")

class ComparisonPlotter:
    def __init__(self, title=None, n_bins=50):
        self.title = title
        self.plot_parts = []
        self.calc_parts = []
        self.components = []

    def add_plot_element(self, element, **kwargs):
        if isclass(element):
            element = element(**kwargs)
        elif isinstance(element, str):
            element_class = REGISTERED_ELEMENTS[element.lower()]
            element = element_class(**kwargs)
        elif isinstance(element, elements.Element):
            pass
        else:
            raise TypeError('Invalid Type \'element\'!')
        logger.info('Adding {}! (Element)'.format(element.name))
        element.register(self)

    def register_calc_part(self, part):
        if not part in self.calc_parts:
            logger.info('\tRegistered {} (CalcPart)!'.format(part.name))
            self.calc_parts.append(part)

    def register_plot_part(self, part):
        if not part in self.calc_parts:
            logger.info('\tRegistered {} (PlotPart)!'.format(part.name))
            self.plot_parts.append(part)

    def add_ref(self,
                label,
                X,
                livetime=1,
                weights=None,
                color=None,
                cmap=None):
        logger.info('Added \'{}\' (Ref-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='ref',
                                         X=X,
                                         weights=weights,
                                         color=color,
                                         cmap=cmap))

    def add_ref_part(self,
                     label,
                     X,
                     livetime=1,
                     weights=None,
                     color=None):
        logger.info('Added \'{}\' (RefPart-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='ref_part',
                                         X=X,
                                         weights=weights,
                                         color=color))

    def add_test(self,
                 label,
                 X,
                 livetime=1,
                 weights=None,
                 color=None):
        logger.info('Added \'{}\' (Test-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='test_part',
                                         X=X,
                                         weights=weights,
                                         color=color))

    def add_test_part(self,
                      label,
                      X,
                      livetime=1,
                      weights=None,
                      color=None):
        logger.info('Added \'{}\' (TestPart-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='test_part',
                                         X=X,
                                         weights=weights,
                                         color=color))

    def draw(self, x_label='Feature', fig=None, figsize=(10, 8)):
        logger.info('Start Draw Process!')
        logger.info('===================')
        result_tray = self.calc()
        if not isinstance(fig, plt.Figure):
            self.last_fig = plt.figure(figsize=figsize)
        total_rows = sum([part_i.rows for part_i in self.plot_parts])
        gs = GridSpec(nrows=total_rows, ncols=1)
        row_point = 0
        logger.info('Starting Plotting...')
        for part_i in self.plot_parts:
            row_slice = slice(row_point, row_pointer + part_i.get_height())
            col_slice = slice(None)
            part_i.set_ax(fig, gs[row_slice, col_slice])
            for comp_i in self.components:
                result_tray = part_i.execute(result_tray, comp_i)
        logger.info('Finished!')
        return fig

    def calc(self):
        logger.info('Starting Calculating...')
        result_tray = ResultTray()
        sorted(self.calc_parts)
        sorted(self.components)
        for part_i in self.calc_parts:
            for comp_i in self.components:
                result_tray = part_i.execute(result_tray, comp_i)
            part_i.reset()
        logger.info('Finished!')
        return result_tray

    def reset(self):
        self.components = []

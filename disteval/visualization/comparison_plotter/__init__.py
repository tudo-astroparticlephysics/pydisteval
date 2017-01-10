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

logger = logging.getLogger("Plotter ")

class ComparisonPlotter:
    def __init__(self, title=None, n_bins=50):
        self.title = title
        self.plot_parts = []
        self.calc_parts = []
        self.components = []
        self.ref_idx = 0

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
        logger.debug('Adding {}! (Element)'.format(element.name))
        element.register(self)

    def register_calc_part(self, part):
        if not part in self.calc_parts:
            logger.debug('\tRegistered {} (CalcPart)!'.format(part.name))
            self.calc_parts.append(part)

    def register_plot_part(self, part):
        if not part in self.plot_parts:
            logger.debug('\tRegistered {} (PlotPart)!'.format(part.name))
            self.plot_parts.append(part)

    def add_ref(self,
                label,
                X,
                livetime=1,
                weights=None,
                color=None,
                cmap=None):
        idx = len(self.components)
        self.ref_idx = idx
        logger.debug('Added \'{}\' (Ref-Component)!'.format(label))
        self.components.append(Component(idx=idx,
                                         label=label,
                                         c_type='ref',
                                         X=X,
                                         livetime=livetime,
                                         weights=weights,
                                         color=color,
                                         cmap=cmap))

    def add_ref_part(self,
                     label,
                     X,
                     livetime=1,
                     weights=None,
                     color=None):
        logger.debug('Added \'{}\' (RefPart-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='ref_part',
                                         X=X,
                                         livetime=livetime,
                                         weights=weights,
                                         color=color))

    def add_test(self,
                 label,
                 X,
                 livetime=1,
                 weights=None,
                 color=None):
        logger.debug('Added \'{}\' (Test-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='test',
                                         X=X,
                                         livetime=livetime,
                                         weights=weights,
                                         color=color))

    def add_test_part(self,
                      label,
                      X,
                      livetime=1,
                      weights=None,
                      color=None):
        logger.debug('Added \'{}\' (TestPart-Component)!'.format(label))
        self.components.append(Component(idx=len(self.components),
                                         label=label,
                                         c_type='test_part',
                                         X=X,
                                         livetime=livetime,
                                         weights=weights,
                                         color=color))

    def draw(self, x_label='Feature', fig=None, figsize=(10, 8)):
        logger.debug('Start Draw Process!')
        logger.debug('===================')
        result_tray = self.calc()
        result_tray.add(x_label, 'x_label')
        if not isinstance(fig, plt.Figure):
            fig = plt.figure(figsize=figsize)
        total_rows = sum([part_i.get_rows() for part_i in self.plot_parts])
        gs = GridSpec(nrows=total_rows, ncols=1)
        row_pointer = 0
        logger.debug('Starting Plotting...')
        for part_i in self.plot_parts:
            part_rows = part_i.get_rows()
            row_slice = slice(row_pointer, row_pointer + part_rows)
            col_slice = slice(None)
            part_i.set_ax(fig, gs[row_slice, col_slice])
            row_pointer += part_rows
            for comp_i in self.components:
                result_tray = part_i.execute(result_tray, comp_i)
            part_i.reset()
        logger.debug('Finished!')
        fig.savefig('test.png')
        return fig

    def calc(self):
        logger.debug('Starting Calculating...')
        result_tray = ResultTray()
        n_components = len(self.components)
        sorted(self.calc_parts)
        sorted(self.components)
        ref_idx = None
        for i, comp in enumerate(self.components):
            comp.idx = i
            if comp.c_type == 'ref':
                if ref_idx is None:
                    ref_idx = i
                else:
                    raise RuntimeError('More than one ref component added!')
        result_tray.add(n_components, 'n_components')
        result_tray.add(ref_idx, 'ref_idx')
        result_tray.add(self.components[ref_idx].livetime, 'ref_livetime')
        for part_i in self.calc_parts:
            for comp_i in self.components:
                result_tray = part_i.execute(result_tray, comp_i)
            part_i.reset()
        logger.debug('Finished!')
        return result_tray

    def reset(self):
        self.components = []

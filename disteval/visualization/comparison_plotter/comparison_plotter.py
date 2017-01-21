# -*- coding: utf-8 -*-
"""Comparison Plotter.

This part of the disteval module is intended to easily produce
high quality plots for the comparison of two distributions. It is
especially designed for automated use and tries to minimize the
knowledge in before hand.
"""
import logging

from matplotlib import pyplot as plt

from . import elements
from .components import Component
from .base_classes import ResultTray, Element

REGISTERED_ELEMENTS = {'aggarwalhisto': elements.AggarwalHisto,
                       'aggarwalratio': elements.AggarwalRatio,
                       'classichisto': elements.ClassicHisto,
                       'classicratio': elements.ClassicRatio,
                       'normalization': elements.Normalization}
logger = logging.getLogger("ComparisonPlotter ")


class ComparisonPlotter:
    """Class to build up the plot layout and produce the plots!

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    title : :obj:`str`
        Title of the plot added to the top of the plot.

    """

    def __init__(self, title=''):
        """Inits an empty comparison plotter.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        title : :obj:`str`m optional (default='')
            Title of the plot..

        """

        self.title = title
        self._plot_parts = []
        self._calc_parts = []
        self._components = []
        self._ref_idx = 0
        self._fig = None

    def add_element(self, element, **kwargs):
        if issubclass(Element):
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
        if part not in self._calc_parts:
            logger.debug('\tRegistered {} (CalcPart)!'.format(part.name))
            self._calc_parts.append(part)

    def register_plot_part(self, part):
        if part not in self._plot_parts:
            logger.debug('\tRegistered {} (PlotPart)!'.format(part.name))
            self._plot_parts.append(part)

    def add_ref(self,
                label,
                X,
                livetime=1,
                weights=None,
                color=None,
                cmap=None):
        idx = len(self._components)
        self._ref_idx = idx
        logger.debug('Added \'{}\' (Ref-Component)!'.format(label))
        self._components.append(Component(idx=idx,
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
        self._components.append(Component(idx=len(self._components),
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
        self._components.append(Component(idx=len(self._components),
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
        self._components.append(Component(idx=len(self._components),
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
            self._fig = plt.figure(figsize=figsize)
        result_tray.add(self._fig, 'fig')
        total_rows = sum([part_i.get_rows() for part_i in self._plot_parts])
        row_pointer = total_rows
        logger.debug('Starting Plotting...')
        ax_dict = {}
        for i, part_i in enumerate(self._plot_parts):
            part_rows = part_i.get_rows()
            y1 = row_pointer / total_rows
            y0 = (row_pointer - part_rows) / total_rows
            x0 = 0.
            x1 = 1.
            part_i.set_ax(fig=self._fig,
                          total_parts=len(self._plot_parts),
                          idx=i,
                          x0=x0,
                          x1=x1,
                          y0=y0,
                          y1=y1)
            row_pointer -= part_rows
            for comp_i in self._components:
                result_tray = part_i.execute(result_tray, comp_i)
            ax_dict[part_i.name] = part_i.get_ax()
            part_i.finish(result_tray)
        logger.debug('Finished!')
        return self._fig, ax_dict, result_tray

    def calc(self):
        logger.debug('Starting Calculating...')
        result_tray = ResultTray()
        n_components = len(self._components)
        self._calc_parts = sorted(self._calc_parts)
        self._components = sorted(self._components)
        ref_idx = None
        test_idx = None
        for i, comp in enumerate(self._components):
            comp.idx = i
            if comp.c_type == 'ref':
                if ref_idx is None:
                    ref_idx = i
                else:
                    raise RuntimeError('More than one ref component added!')
            elif comp.c_type == 'test':
                if test_idx is None:
                    test_idx = i
                else:
                    raise RuntimeError('More than one ref component added!')
        result_tray.add(n_components, 'n_components')
        result_tray.add(ref_idx, 'ref_idx')
        result_tray.add(test_idx, 'test_idx')
        result_tray.add(self._components[ref_idx].livetime, 'ref_livetime')
        result_tray.add(self._components[test_idx].livetime, 'test_livetime')
        for part_i in self._calc_parts:
            for comp_i in self._components:
                result_tray = part_i.execute(result_tray, comp_i)
            part_i.finish(result_tray)
        logger.debug('Finished!')
        return result_tray

    def finish(self):
        plt.close(self._fig)
        self._components = []

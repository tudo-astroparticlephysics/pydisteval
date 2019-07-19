# -*- coding: utf-8 -*-
"""Comparison Plotter.

This part of the disteval module is intended to easily produce
high quality plots for the comparison of two distributions. It is
especially designed for automated use and tries to minimize the
knowledge in before hand.
"""
from __future__ import absolute_import, print_function, division
import logging

from matplotlib import pyplot as plt

from . import elements
from .components import Component, ColorPalette
from .base_classes import ResultTray, Element

REGISTERED_ELEMENTS = {'aggarwalhisto': elements.AggarwalHisto,
                       'aggarwalratio': elements.AggarwalRatio,
                       'limitedmchisto': elements.LimitedMCHisto,
                       'limitedmcratio': elements.LimitedMCRatio,
                       'classichisto': elements.ClassicHisto,
                       'classicratio': elements.ClassicRatio,
                       'normalization': elements.Normalization}
logger = logging.getLogger("ComparisonPlotter ")


class ComparisonPlotter(object):
    """Class to build up the plot layout and produce the plots!

    All parts of the plots are in some way histograms, so continuous
    datapoints discretized in bins. For those bins a Poisson
    distribution is assumed.

    The general concept is that the plot layout is build up with
    'Elements'. Elements consists of 'Parts'. There are two kind of
    parts: 'CalcParts' doing calculations and 'PlotParts' visualizing
    calculations. The layout is setup once. There is no way to remove
    elements once they are added! If you want to change the layout
    just create a new ComparisonPlotter instance.


    After the Layout is created, you have to add components. Components
    mus have at least a name and the data you want to visualize. They
    two mandatory components are the 'test' and the 'ref' components.
    It is not possible to add more than one of those type of components.
    If one of the components consists of multiple contributions that
    should also be plotted, they can be added as 'ref_part's and
    'test_parts'. Those parts and their data are expected to be
    part of the full test/ref component.


    As soon the components are added, the plot are simply drawn with
    the draw() method. To produce the same plot for a different feature
    you have to reset the plotter with the method finish(). The finish
    functions only removes the calculated results, added components
    and resets all the Calc/PlotPart's.

    The process to create the plots is:
    1. Adding elements: (add_element() for each element of the plot)
    for each feature:
        2. Adding Components (add_ref() for the reference component.
                              add_test() for the test component.
                              add_ref_part() for parts of the ref component.
                              add_test_part() for parts of the test component.
        3. Create the plot (draw())
       (4. Reset the plotter, another feature should be plotted (finish())

    Parameters
    ----------
    title : str, optional
        Title of the plot.

    Attributes
    ----------
    title : str
        Title of the plot added to the top of the plot.
    """

    def __init__(self, title=''):
        self.title = title
        self._plot_parts = []
        self._calc_parts = []
        self._components = []
        self._ref_idx = None
        self._test_idx = None
        self._fig = None
        self.color_palette = ColorPalette()

    def add_element(self, element, **kwargs):
        """Method to add a Element.

        The first element added is plotted at the top of the plot.  If parts
        of the element are already registered for the plot, they aren't added
        a second time.

        Note
        ----
        Many elements add the 'CalcBinningPart'. If you want to set
        parameters of the part, you have to do it with the first element
        added.

        Parameters
        ----------
        element :  :obj:`comparison_plotter.base_classes.Element`
            The first parameter.
        **kwarg
            Keyword arguments of the element. See element documentation.

        """
        if isinstance(element, Element):
            element = element(**kwargs)
        elif isinstance(element, str):
            element_class = REGISTERED_ELEMENTS[element.lower()]
            element = element_class(**kwargs)
        elif isinstance(element, elements.Element):
            pass
        else:
            raise TypeError('Invalid Type \'element\'!')
        logger.debug(u'Adding {}! (Element)'.format(element.name))
        element.register(self)

    def _register_calc_part(self, part):
        if part not in self._calc_parts:
            logger.debug(u'\tRegistered {} (CalcPart)!'.format(part.name))
            self._calc_parts.append(part)

    def _register_plot_part(self, part):
        if part not in self._plot_parts:
            logger.debug(u'\tRegistered {} (PlotPart)!'.format(part.name))
            self._plot_parts.append(part)

    def add_ref(self,
                label,
                X,
                livetime=1,
                weights=None,
                color=None,
                cmap=None):
        """Method to add the reference component.

        Parameters
        ----------
        label : str
            Label of the component.
        X : array_like
            Datapoints.
        livetime : float, optional
            Livetime is the time in which the data is taken/produced.
            This is used to be able to proper normalize components of
            different livetime on each other. For this purpose only
            the relative difference between the livetimes is needed.
            So if you have datasets with the same livetime just set
            them all to 1 respectively use the default.
        weights : array_like
            Array of weights. Must be of the same length like X.
        color : matplotlib compatible color code, optional
            A specific color you want this component to have. If None
            a color from the color cycle defined in 'components' module
            is used.
        cmap : matplotlib compatible cmap, optional
            This can be the name of a standard matplotlib colormape.
            Colormaps is mainly used to get ordered colors to indicate
            different confidence levels.

        """
        idx = len(self._components)
        self._ref_idx = idx
        if color is None:
            color = self.color_palette.get_color()
        if cmap is None:
            cmap = self.color_palette.get_cmap()

        logger.debug(u'Added \'{}\' (Ref-Component)!'.format(label))
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
        """Method to add a reference component part.

        Parameters
        ----------
        label : str
            Label of the component.
        X : array_like
            Datapoints.
        livetime : float, optional
            Livetime is the time in which the data is taken/produced.
            This is used to be able to proper normalize components of
            different livetime on each other. For this purpose only
            the relative difference between the livetimes is needed.
            So if you have datasets with the same livetime just set
            them all to 1 respectively use the default.
        weights : array_like
            Array of weights. Must be of the same length like X.
        color : matplotlib compatible color code, optional
            A specific color you want this component to have. If None
            a color from the color cycle defined in 'components' module
            is used.

        """
        logger.debug(u'Added \'{}\' (RefPart-Component)!'.format(label))
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
                 color=None,
                 cmap=None):
        """Method to add a reference component part.

        Parameters
        ----------
        label : str
            Label of the component.
        X : array_like
            Datapoints.
        livetime : float, optional
            Livetime is the time in which the data is taken/produced.
            This is used to be able to proper normalize components of
            different livetime on each other. For this purpose only
            the relative difference between the livetimes is needed.
            So if you have datasets with the same livetime just set
            them all to 1 respectively use the default.
        weights : array_like
            Array of weights. Must be of the same length like X.
        color : matplotlib compatible color code, optional
            A specific color you want this component to have. If None
            a color from the color cycle defined in 'components' module
            is used.
        cmap : matplotlib compatible cmap, optional
            This can be the name of a standard matplotlib colormape.
            Colormaps is mainly used to get ordered colors to indicate
            different confidence levels.

        """
        logger.debug(u'Added \'{}\' (Test-Component)!'.format(label))
        if color is None:
            color = self.color_palette.get_color()
        if cmap is None:
            cmap = self.color_palette.get_cmap()
        self._components.append(Component(idx=len(self._components),
                                          label=label,
                                          c_type='test',
                                          X=X,
                                          livetime=livetime,
                                          weights=weights,
                                          color=color,
                                          cmap=cmap))

    def add_test_part(self,
                      label,
                      X,
                      livetime=1,
                      weights=None,
                      color=None):
        """Method to add a test component part.

        Parameters
        ----------
        label : str
            Label of the component.
        X : array_like
            Datapoints.
        livetime : float, optional
            Livetime is the time in which the data is taken/produced.
            This is used to be able to proper normalize components of
            different livetime on each other. For this purpose only
            the relative difference between the livetimes is needed.
            So if you have datasets with the same livetime just set
            them all to 1 respectively use the default.
        weights : array_like
            Array of weights. Must be of the same length like X.
        color : matplotlib compatible color code, optional
            A specific color you want this component to have. If None
            a color from the color cycle defined in 'components' module
            is used.

        """
        logger.debug(u'Added \'{}\' (TestPart-Component)!'.format(label))
        self._components.append(Component(idx=len(self._components),
                                          label=label,
                                          c_type='test_part',
                                          X=X,
                                          livetime=livetime,
                                          weights=weights,
                                          color=color))

    def draw(self, x_label='Feature', fig=None, figsize=(10, 8)):
        """Method to start the actual draw process.

        In a first step all CalcParts are called for each component.
        In a second step the PlotParts are called for each component.

        Parameters
        ----------
        x_label : str, optional
            Label of the x-axis.
        fig : matplotlib.Figure, optional
            Figure instance that should be used to draw on. If no instance
            is provided a new figure of 'figsize' is created.
        figsize : (width, height), optional
            Tuple of width and height of the figure that is created.

        Returns
        -------
        fig : matplotlib.Figure
            The figure to which the plot is added
        ax_dict : dict
            Dictionary with all the axes created by the PlotParts.
            The key is the name of the PlotPart. If a PlotPart creates
            more than one axis (e.g. AggarwalRatio) a list with all
            axes is added to the dict under the name of the part.

        result_tray : comparison_plotter.base_classes.ResultTray
            A simple object with all the results of the CalcParts as
            the attribute.
        """
        logger.debug(u'Start Draw Process!')
        logger.debug(u'===================')
        result_tray = ResultTray()
        result_tray.add(x_label, 'x_label')
        result_tray = self._calc(result_tray)
        if not isinstance(fig, plt.Figure):
            self._fig = plt.figure(figsize=figsize)
        else:
            self._fig = fig
        result_tray.add(self._fig, 'fig')
        total_rows = sum([part_i.get_rows() for part_i in self._plot_parts])
        row_pointer = total_rows
        logger.debug(u'Starting Plotting...')
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
        logger.debug(u'Finished!')
        return self._fig, ax_dict, result_tray

    def draw_multiples(self, gridsize,
                       x_pos, y_pos,
                       x_label='Feature', fig=None):
        """Method to start the actual draw process.

        In a first step all CalcParts are called for each component.
        In a second step the PlotParts are called for each component.

        Parameters
        ----------
        gridsize : tuple
            Size of the plot grid (Number of rows, number of columns)
        x_pos : int
            Column number to draw to
        y_pos : int
            Row number to draw to
        x_label : str, optional
            Label of the x-axis.
        fig : matplotlib.Figure, optional
            Figure instance that should be used to draw on. If no instance
            is provided a new figure of 'figsize' is created.

        Returns
        -------
        fig : matplotlib.Figure
            The figure to which the plot is added
        ax_dict : dict
            Dictionary with all the axes created by the PlotParts.
            The key is the name of the PlotPart. If a PlotPart creates
            more than one axis (e.g. AggarwalRatio) a list with all
            axes is added to the dict under the name of the part.

        result_tray : comparison_plotter.base_classes.ResultTray
            A simple object with all the results of the CalcParts as
            the attribute.
        """
        logger.debug(u'Start Draw Process!')
        logger.debug(u'===================')
        result_tray = ResultTray()
        result_tray.add(x_label, 'x_label')
        result_tray = self._calc(result_tray)
        self._fig = fig
        result_tray.add(self._fig, 'fig')
        total_rows = sum([part_i.get_rows() for part_i in self._plot_parts])
        row_pointer = total_rows
        logger.debug(u'Starting Plotting...')
        ax_dict = {}
        for i, part_i in enumerate(self._plot_parts):
            part_rows = part_i.get_rows()

            y1_rel = row_pointer / total_rows
            y1_abs = (1 - (y_pos + 1) / gridsize[1]) + \
                y1_rel / gridsize[1]

            y0_rel = (row_pointer - part_rows) / total_rows
            y0_abs = (1 - (y_pos + 1) / gridsize[1]) + \
                y0_rel / gridsize[1]

            x0 = 0. + x_pos / gridsize[0]
            x1 = (x_pos + 1.) / gridsize[0]
            part_i.set_ax(fig=self._fig,
                          total_parts=len(self._plot_parts),
                          idx=i,
                          x0=x0-0.065,
                          x1=x1+0.065,
                          y0=y0_abs,
                          y1=y1_abs,
                          medium_offsets_only=True)
            row_pointer -= part_rows
            for comp_i in self._components:
                result_tray = part_i.execute(result_tray, comp_i)
            ax_dict[part_i.name] = part_i.get_ax()
            part_i.finish(result_tray)
        logger.debug(u'Finished!')
        return self._fig, ax_dict, result_tray

    def _calc(self, result_tray):
        logger.debug(u'Starting Calculating...')
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
        logger.debug(u'Finished!')
        return result_tray

    def finish(self):
        """Method that should be called after drawing.

        It resets the color cycle, removes the components and closes
        the figure. The all added elements are kept.
        """
        logger.debug(u'Finishing: resetting color cycle, component and figure!')
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        self._components = []
        self._ref_idx = None
        self._test_idx = None
        self.color_palette.reset()

    def reset(self, title=''):
        """Method that can be used to do a full reset of the plotter.

        Additional to ComparisonPlotter.finish() also all elements are
        removed. If no title is provided to reset, also the title is emptied.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        """
        logger.debug(u'Reset: resetting parts and components!')
        self.finish()
        self.title = title
        self._plot_parts = []
        self._calc_parts = []

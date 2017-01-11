import logging

import numpy as np

from disteval.visualization.comparison_plotter import ComparisonPlotter

logging.captureWarnings(True)
logging.basicConfig(
        format='%(name)s %(levelname)-8s %(message)s',
        level=logging.DEBUG)

plotter = ComparisonPlotter()
#plotter.add_plot_element('ClassicHisto', bands=False)
#plotter.add_plot_element('ClassicRatio', bands=False, y_lims=[-5, 5])
plotter.add_plot_element('AggarwalHisto')
X_ref = np.random.normal(loc=0., size=10000)
X_test = np.random.normal(loc=0, size=10000)
plotter.add_ref('Reference', X_ref)
plotter.add_test('Test', X_test)
plotter.draw()

import logging

import numpy as np

from disteval.visualization.comparison_plotter import ComparisonPlotter

logging.captureWarnings(True)
logging.basicConfig(
        format='%(name)s %(levelname)-8s %(message)s',
        level=logging.DEBUG)

plotter = ComparisonPlotter()
plotter.add_plot_element('ClassicHisto', bands=False)
plotter.add_plot_element('ClassicRatio', bands=False, y_lims=[-5, 5])
plotter.add_plot_element('AggarwalHisto')
plotter.add_plot_element('Normalization', normalize='test_livetime')
X_ref = np.random.normal(loc=0., size=100000)
X_test = np.random.normal(loc=0, size=1000)
plotter.add_ref('Reference', X_ref, livetime=100)
plotter.add_test('Test', X_test, livetime=1)
plotter.draw()

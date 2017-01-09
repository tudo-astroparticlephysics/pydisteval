import logging

import numpy as np

from disteval.visualization.comparison_plotter import ComparisonPlotter

logging.captureWarnings(True)
logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)

plotter = ComparisonPlotter()
plotter.add_plot_element('ClassicHisto')
X_ref = np.random.normal(loc=1., size=10000)
X_test = np.random.normal(loc=1.1, size=10000)
plotter.add_ref('Reference', X_ref)
plotter.add_test('Test', X_test)
fig = plotter.draw()

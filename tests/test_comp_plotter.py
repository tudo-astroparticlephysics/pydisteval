import logging

import numpy as np

from disteval.visualization.comparison_plotter import ComparisonPlotter

logging.captureWarnings(True)
logging.basicConfig(
        format='%(name)s %(levelname)-8s %(message)s',
        level=logging.DEBUG)

plotter = ComparisonPlotter()


plotter.add_element('AggarwalHisto', n_bins=50, alpha=[0.68, 0.9, 0.99])
plotter.add_element('AggarwalRatio', zoomed=True)
plotter.add_element('Normalization', normalize='test_livetime')

for i in range(1):
    X_ref = np.random.normal(loc=0., size=100000)
    X_test = np.random.normal(loc=0.1, size=100000)
    plotter.add_ref('Sum MC', X_ref, livetime=100, color='k')
    plotter.add_test('Data', X_test, livetime=100)

    fig, ax_dict, _ = plotter.draw('x')
    fig.savefig('test{}.png'.format(i))
    plotter.finish()

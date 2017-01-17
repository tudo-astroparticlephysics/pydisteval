import logging

import numpy as np

from disteval.visualization.comparison_plotter import ComparisonPlotter

logging.captureWarnings(True)
logging.basicConfig(
        format='%(name)s %(levelname)-8s %(message)s',
        level=logging.DEBUG)

plotter = ComparisonPlotter()


plotter.add_element('AggarwalHisto')
plotter.add_element('AggarwalRatio')
plotter.add_element('Normalization', normalize='test_livetime')



for i in range(10):
    X_ref = np.random.normal(loc=0., size=100000)
    X_test = np.random.normal(loc=0, size=1000)
    plotter.add_ref('Sum MC', X_ref, livetime=100, color='k')
    plotter.add_ref_part('Corsika', X_ref[:3000], livetime=100)
    plotter.add_ref_part('Corsika', X_ref[3000:4000], livetime=100)
    plotter.add_test('Data', X_test, livetime=1)

    fig, ax_dict, _ = plotter.draw()
    fig.savefig('test{}.png'.format(i))
    plotter.finish()

import logging

import numpy as np

from sklearn.datasets import make_regression

from disteval.discretization import TreeBinningSklearn


if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)
    n_samples = 10000
    X, y = make_regression(n_samples=n_samples)

    idx = int(0.5 * len(y))
    X_test = X[idx:, :]
    X_train = X[:idx, :]
    y_train= y[:idx]
    logging.info('Binning with at least 100 samples per bin and not more '
                 'than 10 leafs.')
    clf = TreeBinningSklearn(
        regression=True,
        min_samples_leaf=100,
        max_leaf_nodes=10,
        random_state=1337)
    clf.fit(X_train, y_train)
    binned_X_test = clf.digitize(X_test)
    logging.info('Histogram of the binned test sample:')
    logging.info(np.bincount(binned_X_test))

    logging.info('Binning with at least 100 samples per bin and no limit on '
                 'the number of leafs.')
    clf = TreeBinningSklearn(
        regression=True,
        min_samples_leaf=100,
        #max_leaf_nodes=10,
        random_state=1337)
    clf.fit(X_train, y_train)
    binned_X_test = clf.digitize(X_test)
    logging.info('Histogram of the binned test sample:')
    logging.info(np.bincount(binned_X_test))

    logging.info('Using the test sample to prune the tree and ensure at '
                 'least 100 events in each bin:')
    clf.prune(X_test, 100)
    binned_X_test = clf.digitize(X_test)
    logging.info('Histogram of the pruned binning:')
    logging.info(np.bincount(binned_X_test))

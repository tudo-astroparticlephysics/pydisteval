from disteval.discretization import TreeBinningSklearn
import numpy as np
from sklearn.datasets import make_regression

from matplotlib import pyplot as plt


def test_treebinning():
    n_samples = 1000
    X, y = make_regression(n_samples=n_samples)
    plt.hist(y)
    y_binned = np.digitize(y, np.linspace(-400, 400, 20))
    idx = int(0.9 * n_samples)
    X_test = X[idx:, :]
    X_train = X[:idx, :]
    y_train = y_binned[:idx]
    clf = TreeBinningSklearn(
        regression=True,
        max_features=None,
        min_samples_split=2,
        max_depth=None,
        min_samples_leaf=100,
        max_leaf_nodes=10,
        random_state=1337)
    clf.fit(X_train, y_train)
    score = clf.predict(X_test)
    assert len(score) == X_test.shape[0]
    leaves_test = clf.digitize(X_test)
    leaves_train = clf.digitize(X_train)
    assert len(leaves_test) == X_test.shape[0]
    hist_test = np.bincount(leaves_test)
    hist_train = np.bincount(leaves_train)

    assert len(hist_train) <= 10
    assert min(hist_train) >= 100
    min_test = min(hist_test) + 1
    n_below_threshold = sum(hist_test < min_test)
    clf.prune(X_test, min_test)
    leaves_test = clf.digitize(X_test)
    assert len(leaves_test) == X_test.shape[0]
    hist_test_pruned = np.bincount(leaves_test)
    assert all(hist_test_pruned > min_test)
    assert len(hist_test_pruned) == len(hist_test) - n_below_threshold
    assert sum(hist_test_pruned) == sum(hist_test)

if __name__ == '__main__':
    import logging
    logging.captureWarnings(True)
    logging.basicConfig(
        format='%(processName)-10s %(name)s %(levelname)-8s %(message)s',
        level=logging.INFO)
    test_treebinning()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import logging

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import disteval
from disteval import evaluation as eval

log = logging.getLogger("disteval.fact_example")

def main():
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s|%(name)s|%(levelname)s| ' +
                        '%(message)s'), level=logging.INFO)
    log.info("Starting ROC comparison example with FACT data")

    X, y = make_classification(n_samples=10000, n_features=20,
                               n_informative=2, n_redundant=10,
                               random_state=42)
    sample_weight = None
    alpha = 0.05

    clf = RandomForestClassifier(n_jobs=40, n_estimators=200)

    log.info("Real classifiaction")
    clf, y_pred_a, cv_step = disteval.cv_test_ref_classification(
        clf, X, y, sample_weight, cv_steps=10, return_all_models=False)

    log.info("Classifiaction with random label (Guessing clf)")
    y_fake_b = np.random.randint(0, 2, size=X.shape[0])
    clf, y_pred_b, cv_step = disteval.cv_test_ref_classification(
        clf, X, y_fake_b, sample_weight, cv_steps=10, return_all_models=False)
    log.info("Running ROC curve equivalence test")
    roc_eval_result = eval.roc_curve_equivalence_ks_test(y_pred_a,
                                                         y_pred_b,
                                                         y,
                                                         alpha=alpha)
    if roc_eval_result[0]:
        log.info("The test was not rejected!")
    else:
        log.info("The test was rejected! (alpha={})".format(alpha))

    plt.plot(roc_eval_result[3], roc_eval_result[4], 'r-', label='Correct')
    plt.plot(roc_eval_result[5], roc_eval_result[6], 'k--', label='Guessing')
    plt.plot(roc_eval_result[1][0, :], roc_eval_result[1][1, :], 'o')
    plt.plot(roc_eval_result[2][0, :], roc_eval_result[2][1, :], 'x')
    plt.legend(loc=4)
    plt.savefig('test1.png')
    plt.clf()
    plt.plot(roc_eval_result[7], roc_eval_result[3], 'r--',
             label='FPR Real Classification')
    plt.plot(roc_eval_result[7], roc_eval_result[5], 'k--',
             label='FPR Guessing Classification')
    plt.plot(roc_eval_result[7], roc_eval_result[4], 'r-',
             label='TPR Real Classification')
    plt.plot(roc_eval_result[7], roc_eval_result[6], 'k-',
             label='TPR Guessing Classification')
    plt.legend(loc=4)
    plt.savefig('rates.png')
    plt.clf()

    log.info("Second guessing classifier is trained")
    y_fake_c = np.random.randint(0, 2, size=X.shape[0])
    clf, y_pred_c, cv_step = disteval.cv_test_ref_classification(
        clf, X, y_fake_c, sample_weight, cv_steps=10, return_all_models=False)
    log.info("Running ROC curve equivalence test for both guessing clf")
    roc_eval_result = eval.roc_curve_equivalence_ks_test(y_pred_b,
                                                         y_pred_c,
                                                         y_fake_b,
                                                         alpha=alpha)
    if roc_eval_result[0]:
        log.info("The test was not rejected!")
    else:
        log.info("The test was rejected! (alpha={})".format(alpha))
    plt.plot(roc_eval_result[3], roc_eval_result[4], 'r-', label='Guessing 1')
    plt.plot(roc_eval_result[5], roc_eval_result[6], 'k--', label='Guessing 2')
    plt.plot(roc_eval_result[1][0, :], roc_eval_result[1][1, :], 'o')
    plt.plot(roc_eval_result[2][0, :], roc_eval_result[2][1, :], 'x')
    plt.legend(loc=4)
    plt.savefig('test2.png')
    plt.clf()

    log.info("Running ROC curve test with the correponding faked truth values")
    roc_eval_result = eval.roc_curve_equivalence_ks_test(y_pred_b,
                                                         y_pred_c,
                                                         y_fake_b,
                                                         y_fake_c,
                                                         alpha=alpha)
    if roc_eval_result[0]:
        log.info("The test was not rejected!")
    else:
        log.info("The test was rejected! (alpha={})".format(alpha))
    plt.plot(roc_eval_result[3], roc_eval_result[4], 'r-', label='Guessing 1')
    plt.plot(roc_eval_result[5], roc_eval_result[6], 'k--', label='Guessing 2')
    plt.plot(roc_eval_result[1][0, :], roc_eval_result[1][1, :], 'o')
    plt.plot(roc_eval_result[2][0, :], roc_eval_result[2][1, :], 'x')
    plt.legend(loc=4)
    plt.savefig('test3.png')
    plt.clf()


if __name__ == "__main__":
    main()

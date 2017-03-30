# -*- coding:utf-8 -*-
'''Example usage of the 'roc_curve_equivalence_ks_test' and the
'visualize_roc_curve_equivalence_test'
'''
import logging
import matplotlib
matplotlib.use('Agg')
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import disteval
from disteval import evaluation as eval
from disteval import visualization as visu

log = logging.getLogger("disteval.roc_curve_equivalence_example")



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
    y_fake = np.random.randint(0, 2, size=X.shape[0])
    clf, y_pred_guess, cv_step = disteval.cv_test_ref_classification(
        clf, X, y_fake, sample_weight, cv_steps=10, return_all_models=False)
    log.info("Running ROC curve equivalence test")
    roc_eval_result = eval.roc_curve_equivalence_ks_test(y_pred_a,
                                                         y_pred_guess,
                                                         y,
                                                         alpha=alpha)
    if roc_eval_result[0]:
        log.info("The test was not rejected!")
    else:
        log.info("The test was rejected! (alpha={})".format(alpha))

    visu.visualize_roc_curve_equivalence_test(roc_eval_result,
                                              save_path='real_guessing.png')


if __name__ == "__main__":
    main()

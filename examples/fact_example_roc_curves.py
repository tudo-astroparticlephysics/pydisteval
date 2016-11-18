import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import logging

from sklearn.ensemble import RandomForestClassifier

import disteval
from disteval import evaluation as eval

log = logging.getLogger("disteval.fact_example")

test_filename1 = '/fhgfs/groups/app/fact/data_analysis_output/facttoolsParameterRootFiles/AnalysisV_sourceFix/Crab.hdf5'

test_filename2 = '/fhgfs/groups/app/fact/simulated/FacttoolsParamRootFiles/AnalysisV_sourceFix/proton_12.hdf5'

training_variables = ['ConcCore',
                      'Concentration_onePixel',
                      'Concentration_twoPixel',
                      'Leakage',
                      'Leakage2',
                      'Size',
                      'Slope_long',
                      'Slope_spread',
                      'Slope_spread_weighted',
                      'Slope_trans',
                      'Distance',
                      'Theta',
                      'Timespread',
                      'Timespread_weighted',
                      'Width',
                      'arrTimePosShower_kurtosis',
                      'arrTimePosShower_max',
                      'arrTimePosShower_mean',
                      'arrTimePosShower_min',
                      'arrTimePosShower_skewness',
                      'arrTimePosShower_variance',
                      'arrTimeShower_kurtosis',
                      'arrTimeShower_max',
                      'arrTimeShower_mean',
                      'arrTimeShower_min',
                      'arrTimeShower_skewness',
                      'arrTimeShower_variance',
                      'concCOG',
                      'm3l',
                      'm3t',
                      'maxPosShower_kurtosis',
                      'maxPosShower_max',
                      'maxPosShower_mean',
                      'maxPosShower_min',
                      'maxPosShower_skewness',
                      'maxPosShower_variance',
                      'maxSlopesPosShower_kurtosis',
                      'maxSlopesPosShower_max',
                      'maxSlopesPosShower_mean',
                      'maxSlopesPosShower_min',
                      'maxSlopesPosShower_skewness',
                      'maxSlopesPosShower_variance',
                      'maxSlopesShower_kurtosis',
                      'maxSlopesShower_max',
                      'maxSlopesShower_mean',
                      'maxSlopesShower_min',
                      'maxSlopesShower_skewness',
                      'maxSlopesShower_variance',
                      'numIslands',
                      'numPixelInShower',
                      'phChargeShower_kurtosis',
                      'phChargeShower_max',
                      'phChargeShower_mean',
                      'phChargeShower_min',
                      'phChargeShower_skewness',
                      'phChargeShower_variance',
                      'photonchargeMean',
                      ]


def main():
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s|%(name)s|%(levelname)s| ' +
                        '%(message)s'), level=logging.INFO)
    log.info("Starting ROC comparison example with FACT data")

    data_df = pd.read_hdf(test_filename1)
    mc_df = pd.read_hdf(test_filename2)
    alpha = 0.05


    log.info("Reducing Features")
    data_df = data_df.loc[:10000, training_variables]
    mc_df = mc_df.loc[:, training_variables]

    clf = RandomForestClassifier(n_jobs=40, n_estimators=200)

    log.info("Data preparation")
    X, y, sample_weight, X_names = disteval.prepare_data(mc_df,
                                                         data_df,
                                                         test_weight=None,
                                                         ref_weight=None,
                                                         test_ref_ratio=1.,
                                                         )
    del data_df
    del mc_df

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

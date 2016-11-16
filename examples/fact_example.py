import numpy as np
import pandas as pd

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
    log.info("Starting FACT example")

    data_df = pd.read_hdf(test_filename1)
    mc_df = pd.read_hdf(test_filename2)

    log.info("Reducing Features")
    data_df = data_df.loc[:, training_variables]
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

    log.info("test classifiaction")
    clf, y_pred, cv_step = disteval.cv_test_ref_classification(
        clf, X, y, sample_weight, cv_steps=10, return_all_models=True)

    kept, mean_imp, std_imp = eval.feature_importance_mad(clf, alpha=0.05)
    removed_features_str = ''
    for i in np.argsort(mean_imp)[::-1]:
        if not kept[i]:
            removed_features_str += '{}, '.format(X_names[i])

    log.info("Removed Features MAD evaluation:")
    log.info("[Order from high to low mean importance]")
    log.info(removed_features_str)

    kept, mean_imp, std_imp = eval.feature_importance_mad_majority(
        clf, ratio=0.9, alpha=0.10)
    removed_features_str = ''
    for i in np.argsort(mean_imp)[::-1]:
        if not kept[i]:
            removed_features_str += '{}, '.format(X_names[i])
    log.info("Removed Features majority MAD evaluation:")
    log.info("[Order from high to low mean importance]")
    log.info(removed_features_str)

    clf = RandomForestClassifier(n_jobs=10, n_estimators=50)

    selected_features = disteval.recursive_feature_selection_roc_auc(
        clf,
        X,
        y,
        n_features=10,
        cv_steps=5,
        n_jobs=4,
        forward=True,
        matching_features=False)

    removed_features_str = ''
    for i in selected_features:
        removed_features_str += '{}, '.format(X_names[i])
    log.info("Features obtain via Forward Selection:")
    log.info("[Order from early to late selection]")
    log.info(removed_features_str)


if __name__ == "__main__":
    main()

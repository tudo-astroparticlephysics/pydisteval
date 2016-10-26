
def main():
    """Entry point for the application script"""
    print("Call your main application code here")


def roc_mismatch(test_df, ref_df, clf):
    """Runs a classification betwenn the test data and the reference data.
    For this classification the ROC-Curve  is analysed to check if the
    classifier is sensitive for potential mismathces.
    The hypothesis for the analyse is that the test data has the same
    distribtuion as the reference data.

    Parameters
    ----------
    test_df : pandas.Dataframe, shape=(n_samples_mc, features)
        Dataframe of the test data

    ref_df : pandas.Dataframe, shape=(n_samples_mc, features)
        Dataframe of the reference data

    Returns
    -------
    ???
    """
    raise NotImplementedError

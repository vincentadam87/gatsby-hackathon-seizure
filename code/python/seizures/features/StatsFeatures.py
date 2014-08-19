import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.Instance import Instance
import scipy.stats as st

class StatsFeatures(FeatureExtractBase):
    """
    Bunch of basic statistics features e.g., kurtosis, coefficient of variation.
    Very simple but may be informative in some sense.

    See scipy's /stats.html#statistical-functions

    @author Wittawat 
    """

    def __init__(self):
        pass

    def extract(self, instance):
        assert(isinstance(instance, Instance))
        dndata = instance.eeg_data

        kurtosis = st.kurtosis(dndata, axis=1)
        skew = st.skew(dndata, axis=1)
        # coefficient of variation 
        variation = st.variation(dndata, axis=1)
        
        # hstack will collapse all entries into one big vector 
        features = np.hstack( (kurtosis, skew, variation) )
        self.assert_features(features)
        # features = a 1d ndarray 
        return features

    def __str__(self):
        return "Stats"




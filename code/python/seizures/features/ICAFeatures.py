import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.Instance import Instance
import scipy.stats as st

class ICAFeatures(FeatureExtractBase):
    """
    See http://scot-dev.github.io/scot-doc/api/scot/scot.html#scot.plainica.plainica

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




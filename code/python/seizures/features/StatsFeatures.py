import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.Instance import Instance
import scipy.stats as st

class StatsFeatures(FeatureExtractBase):
    """
    Bunch of basic statistics features e.g., kurtosis, coefficient of variation.
    Very simple but may be informative in some sense.
    I don't expect it to be so useful. Just wonder how discriminative these 
    features are. 

    From simple testing, these features alone give 85% auc on seizure task and 
    72% on early seizure task on Dog_1 data. Better than coin flips ?

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
        Inan = np.where(np.isnan(features))
        Iinf = np.where(np.isinf(features))
        features[Inan] = 0
        features[Iinf] = 0

        self.assert_features(features)
        # features = a 1d ndarray 
        return np.hstack(features)

    def __str__(self):
        return "Stats"




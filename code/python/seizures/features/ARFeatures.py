import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from statsmodels.tsa.vector_ar.var_model import VAR

class ARFeatures(FeatureExtractBase):
    """
    Class to extracts AR(2) features.
    @author V&J
    """

    def __init__(self):
        pass

    def extract(self, instance):
        # instance is an object of class Instance
        params = VAR(instance.eeg_data.T).fit(maxlags=2).params
        # Wittawat: Do we need np.hstack(.) ? 
        features = np.hstack(params.reshape( (np.prod(params.shape),1) ))
        self.assert_features(features)
        # features = a 1d ndarray 
        return features

import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from statsmodels.tsa.vector_ar.var_model import VAR
#from code.python.seizures.data.Instance import Instance

class ARFeatures(FeatureExtractBase):
    """
    Class to extracts AR(2) features.
    @author V&J
    """

    def __init__(self):
        pass

    def extract(self, instance):
        # instance is an object of class Instance

        # Wittawat: Since VAR automatically does lags order selection, 
        # other different instances may give a different lags values ?
        #print instance.eeg_data.T.shape, instance.eeg_data.T
        #v= VAR(instance.eeg_data.T)

        params = VAR(instance.eeg_data.T)._estimate_var(1).params
        features = np.hstack(params.reshape( (np.prod(params.shape), 1) ))
        self.assert_features(features)
        # features = a 1d ndarray 
        return features

    def __str__(self):
        return "AR"

# ----- end of ARFeatures ------------

class VarLagsARFeatures(FeatureExtractBase):
    """
    A feature generator which uses autoregressive coefficients as the features.
    The lags parameter can be specified. Each lags value yields a different 
    set of features. 

    Features obtained from multiple lags values may be conbined with 
    MixFeatures or seizures.features.StackFeatures.

    @author Wittawat 
    """

    def __init__(self, lags):
        # lags to be used when finding autoregressive coefficients
        # expect an integer.
        self.lags = lags

    def extract(self, instance):
        #assert(isinstance(instance, Instance))
        #print self.lags, instance.eeg_data.T.shape, instance.eeg_data.T
        #v= VAR(instance.eeg_data.T)
        params = VAR(instance.eeg_data.T)._estimate_var(self.lags).params
        # hstack will collapse all entries into one big vector 
        features = np.hstack(params.reshape( (np.prod(params.shape),1) ))
        self.assert_features(features)
        # features = a 1d ndarray 
        return features

    def __str__(self):
        return 'VLAR' + '(%d)'% (self.lags)




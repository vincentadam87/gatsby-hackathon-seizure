import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from statsmodels.tsa.vector_ar.var_model import VAR
# dependency : http://statsmodels.sourceforge.net/

class ARFeatures(FeatureExtractBase):
    """
    Class to extracts AR(2) features.
    @author V&J
    """

    def __init__(self):
        pass

    def extract(self, instance):
        params = VAR(instance.eeg_data.T).fit(maxlags=2).params
        return params.reshape( (np.prod(params.shape),1) )
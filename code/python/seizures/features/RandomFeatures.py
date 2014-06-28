from abc import abstractmethod

import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase


class RandomFeatures(FeatureExtractBase):
    """"
    Class that generates random features, ignoring the raw data.
    This is to make the interfaces work.
    
    @author: Heiko
    """

    @abstractmethod
    def extract(self, eeg_data_instance):
        """
        Returns random feature vector of dimension 10
        
        Returns:
        A 1d numpy array with random features
        """
        return np.random.randn(10)
    

from abc import abstractmethod
import numpy as np

class FeatureExtractBase(object):
    """"
    Abstract base class that implement feature extraction interface.

    @author: Heiko
    """

    @abstractmethod
    def extract(self, eeg_data_instance):
        """
        Method to extract features in whatever way

        Returns:
        A 2d numpy array with features.
        """
        raise NotImplementedError()

    def assert_features(self, features):
        assert(type(features)==np.ndarray)
        assert(len(features.shape)==1)
        assert(features.shape[0]>=1)

    def __str__(self):
        # subclass may override this. Be sure to make it readable.
        return type(self).__name__.split('.')[-1]

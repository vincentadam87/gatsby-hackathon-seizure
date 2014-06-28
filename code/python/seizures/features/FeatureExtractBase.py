from abc import abstractmethod


class FeatureBase(object):
    """"
    Abstract base class that implement feature extraction interface.
    
    @author: Heiko
    """

    @abstractmethod
    def extract(self):
        """
        Method to extract features in whatever way
        
        Returns:
        A 2d numpy array with features and a 1d numpy array with labels.
        """
        raise NotImplementedError()

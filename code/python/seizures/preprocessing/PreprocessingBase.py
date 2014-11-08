from abc import abstractmethod

class PreprocessingBase(object):
    """"
    Abstract base class to declare a preprocessing method.
    @author: Vincent
    """

    @abstractmethod
    def apply(self, X):
        """
        Method to apply preprocessing to signals
        :param X: 2D np.ndarray the unprocessed data
        :param X: 2D np.ndarray the processed data
        """
        raise NotImplementedError()

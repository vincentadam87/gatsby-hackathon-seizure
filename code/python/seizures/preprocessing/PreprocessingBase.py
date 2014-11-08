from abc import abstractmethod

class PreprocessingBase(object):
    """"
    Abstract base class to declare a preprocessing method.
    @author: Vincent
    """

    @abstractmethod
    def apply(self, X, fs):
        """
        Method to apply preprocessing to signals
        :param X: 2D np.ndarray the unprocessed data
        :param fs: sampling frequency
        :return: 2D np.ndarray the processed data
        """
        raise NotImplementedError()

    @abstractmethod
    def applyToPatientData(self, P):
        """
        Method to apply preprocessing to signals
        :param P: an instance of PatientData
        :return: PatientData
        """
        raise NotImplementedError()


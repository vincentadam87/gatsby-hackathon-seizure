__author__ = 'nuke'
from seizures.preprocessing.PreprocessingBase import PreprocessingBase
from abc import abstractmethod

class PreprocessingChain(PreprocessingBase):
    def __init__(self, preprocessingList):
        self.preprocessingList = preprocessingList


    @abstractmethod
    def apply(self, x):
        """
        method to apply preprocessing to signals
        :param x: 2d np.ndarray the unprocessed data
        :return: 2d np.ndarray the processed data
        """
        raise NotImplementedError()

    @abstractmethod
    def applytopatientdata(self, p):
        """
        method to apply preprocessing to signals
        :param p: an instance of patientdata
        :return: patientdata
        """
        raise NotImplementedError()



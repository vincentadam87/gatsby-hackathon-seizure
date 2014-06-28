from abc import abstractmethod


class PredictorBase(object):
    """"
    Abstract base class that implement the interface that we use for our
    predictors. Classic supervised learning.
    
    @author: Heiko
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Method to fit the model.
        
        Parameters:
        X - 2d numpy array of training data
        y - 1d numpy array of training labels
        """
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, X):
        """
        Method to apply the model data
        
        Parameters:
        X - 2d numpy array of test data
        """
        raise NotImplementedError()

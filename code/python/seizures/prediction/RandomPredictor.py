from abc import abstractmethod

import numpy as np
from seizures.prediction.PredictorBase import PredictorBase


class RandomPredictor(PredictorBase):
    """"
    Abstract base class that implement the interface that we use for our
    predictors. Classic supervised learning.

    @author: Heiko
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Does nothing

        Parameters:
        X - 2d numpy array of training data
        y - 1d numpy array of training labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Returns random predictions in (0,1)

        Parameters:
        X - 2d numpy array of test data
        """
        return np.random.rand(len(X))

from abc import abstractmethod
import numpy as np

class RandomPredictor(object):
    """"
    Abstract base class that implement the interface that we use for our
    predictors. Classic supervised learning.

    @author: Heiko
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Method to fit the model. In this case, nothing to do.

        Parameters:
        X - 2d numpy array of training data
        y - 1d numpy array of training labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Method to apply the model data

        Parameters:
        X - 2d numpy array of test data
        """
        return (hash(str(X)) % 100) / 100.0

if __name__ == '__main__':
    X = np.random.rand(100, 100)

    predictor = RandomPredictor()
    y1 = predictor.predict(X)
    y2 = predictor.predict(X)

    assert(y1 == y2)
    assert(y1 > 0.0 and y2 < 1.0)
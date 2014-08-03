from abc import abstractmethod
import numpy as np
from sklearn.svm import SVC


class SVMPredictor(object):
    """"
    A simple application of SVM classifier

    @author: Shaun
    """

    def __init__(self):
        self.clf = SVC(probability=True)

    @abstractmethod
    def fit(self, X, y):
        """
        Method to fit the model.

        Parameters:
        X - 2d numpy array of training data
        y - 1d numpy array of training labels
        """
        self.clf = self.clf.fit(X, y)

    @abstractmethod
    def predict(self, X):
        """
        Method to apply the model data

        Parameters:
        X - 2d numpy array of test data
        """
        return self.clf.predict_proba(X)[:, 1]

if __name__ == '__main__':
    N = 1000
    D = 2
    X = np.random.rand(N, D)
    y = np.random.randint(0, 2, N)

    predictor = SVMPredictor()
    predictor.fit(X, y)

    x = np.random.rand(1, D)
    pred = predictor.predict(x)

    print pred





from abc import abstractmethod

import numpy as np
from seizures.prediction.PredictorBase import PredictorBase
from sklearn.ensemble import ExtraTreesClassifier


class XtraTreesPredictor(PredictorBase):
    """"
    A simple application of RandomForestClassifier

    @author: Shaun
    """

    def __init__(self, n_estimators=100, max_features='auto'):
        self.clf = ExtraTreesClassifier(n_estimators=n_estimators,
                max_features=max_features)

    @abstractmethod
    def fit(self, X, y):
        """
        Method to fit the model.

        Parameters:
        X - 2d numpy array of training data. X.shape = [n_samples, d_features]
        y - 1d numpy array of training labels
        """
        print "Fitting a random forest predictor"
        self.clf = self.clf.fit(X, y)

    @abstractmethod
    def predict(self, X):
        """
        Method to apply the model data

        Parameters:
        X - 2d numpy array of test data
        """
        # [:, 1] to get the second column, which contains the probabilies of
        # of class being 1
        return self.clf.predict_proba(X)[:, 1]

    def __str__(self):
        return "Forest"

if __name__ == '__main__':
    N = 1000
    D = 2
    X = np.random.rand(N, D)
    y = np.random.randint(0, 2, N)

    predictor = ExtraTreesClassifier()
    predictor.fit(X, y)

    x = np.random.rand(1, D)
    pred = predictor.predict(x)

    print pred





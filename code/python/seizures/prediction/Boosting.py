from abc import abstractmethod

import numpy as np
from seizures.prediction.PredictorBase import PredictorBase
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class AdaBoostTrees(PredictorBase):
    """"
    AdaBoost + Decision trees.
    See http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html 

    @author: Wittawat
    """

    def __init__(self, **options):
        """
        options is a dictionary to be used as arguments to DecisionTreeClassifier.
        """
        # No strong justification why max_depth=5. Change if needed.
        self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, **options),
                algorithm="SAMME",
                n_estimators=100 )

    def fit(self, X, y):
        """
        Parameters:
        X - 2d numpy array of training data. X.shape = [n_samples, d_features]
        y - 1d numpy array of training labels
        """
        print "fitting AdaBoost trees"
        self.clf = self.clf.fit(X, y)

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
        return "ABTrees"

def main():
    N = 399 
    D = 20
    X = np.random.rand(N, D)
    y = np.random.randint(0, 2, N)

    predictor = AdaBoostTrees(max_features=10)
    predictor.fit(X, y)

    x = np.random.rand(1, D)
    pred = predictor.predict(x)

    print pred

if __name__ == '__main__':
    main()


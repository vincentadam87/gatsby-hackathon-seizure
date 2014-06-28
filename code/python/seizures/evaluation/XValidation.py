from sklearn.cross_validation import StratifiedShuffleSplit

import numpy as np


class XValidation():
    """
    Class to provide basic cross-validation for a generic supervised prediction
    method which has an interface that takes a a bunch of feature vectors
    (as a matrix) and labels for training and then offers a predict method that
    takes a bunch of new vectors to predict a label for.
    """
    
    @staticmethod
    def evaluate(X, y, test_size=0.1, n_iter=1, prediction, evaluation):
        """
        Performs stratified cross-validation on training data X and labels y.
        Assumes that y is discrete.
        
        Parameters:
        X          - training data, 2d numpy array
        y          - training labels, 1d numpy array
        test_size  - number on (0,1) denoting fraction of data used for testing
        n_iter     - number of repetitions (i.e. x-validation runs)
        prediction - instance of PredictorBase
        evaluation - function handle that takes two equally sized 1d vectors
                     and evaluates some performance measure on them.
                     
        Returns:
        2d array where each row corresponds to the performance measure on test
        set for each fold.
        
        @author: Heiko
        """
        # make sure we get right types
        assert(type(X) == np.ndarray)
        assert(type(y) == np.ndarray)
        
        # array sizes
        assert(len(X.shape) == 2)
        assert(len(y.shape) == 1)
        
        # array dimensions
        assert(X.shape[1] == len(y))
        assert(X.shape[0] > 0)
        
        # make sure there is more than one class
        assert(len(np.unique(y))>1)
        
        # create stratified iterator sets using sklearn
        sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=test_size)
        
        # run x-validation
        result = []
        for train_index, test_index in sss:
            # partition data
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            
            # run predictor
            y_predict = prediction.fit(X_train, y_train)
            y_test = prediction.predict(X_test)
            
            # some sanity checks on the provided predictor to avoid problems
            if not type(y_predict) == np.ndarray:
                raise TypeError("Provided predictor doesn't return numpy array")
            
            if not len(y_predict.shape) == 1:
                raise TypeError("Provided predictor doesn't not return 1d array")
            
            if not len(y_predict) == len(y_test):
                raise TypeError("Provided predictor doesn't return right number of labels")
            
            # evaluate, store
            score = evaluation(X_test, y_predict)
            result.append(score)
        
        # return as 2d array
        return np.asarray(result).reshape(n_iter, len(result)/n_iter)
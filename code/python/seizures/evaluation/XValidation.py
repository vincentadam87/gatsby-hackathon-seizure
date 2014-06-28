from sklearn.cross_validation import StratifiedShuffleSplit

import numpy as np
from seizures.evaluation.auc import auc


class XValidation():
    """
    Class to provide basic cross-validation for a generic supervised prediction
    method which has an interface that takes a a bunch of feature vectors
    (as a matrix) and labels for training and then offers a predict method that
    takes a bunch of new vectors to predict a label for.
    
    Note that training data here has a special structure: list of 2d arrays,
    see evaluate method.
    """
    
    @staticmethod
    def evaluate(X_list, y, predictor, test_size=0.1, n_iter=1, evaluation=auc):
        """
        Performs stratified cross-validation on training data X and labels y.
        Assumes that y is discrete.
        
        Note that the training data is in the form of a list of matrices,
        where each matrix contains feature vectors for one particular seizure.
        Cross-validation is done over those matrices, the classifier is then 
        trained on the concatenated matrices within those blocks.
        
        Parameters:
        X_list     - training data, list of 2d numpy arrays
        y          - training labels, 1d numpy array, same length as above list
        predictor  - instance of PredictorBase
        test_size  - number on (0,1) denoting fraction of data used for testing
        n_iter     - number of repetitions (i.e. x-validation runs)
        evaluation - function handle that takes two equally sized 1d vectors
                     and evaluates some performance measure on them.
                     Optional, default is AUC
                     
        Returns:
        1d array where each entry corresponds to the performance measure the 
        test folds (n_iter many)
        
        @author: Heiko
        """
        # make sure we get right types
        assert(type(y) == type(list))
        for X in X_list:
            assert(type(X) == np.ndarray)
            
        assert(type(y) == np.ndarray)
        assert(type(test_size) == float)
        
        # array sizes
        for X in X_list:
            assert(len(X) == 2)
            assert(X.shape[0] > 0)
            
        assert(len(y.shape) == 1)
        
        # number of list elements and labels
        assert(len(y) == len(y))
        
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
            predictor.fit(X_train, y_train)
            y_predict = predictor.predict(X_test)
            
            # some sanity checks on the provided predictor to avoid problems
            if not type(y_predict) == np.ndarray:
                raise TypeError("Provided predictor doesn't return numpy array, but %s"%str(type(y_predict)))
            
            if not len(y_predict.shape) == 1:
                raise TypeError("Provided predictor doesn't not return 1d array")
            
            if not len(y_predict) == len(y_test):
                raise TypeError("Provided predictor doesn't return right number of labels")
            
            # evaluate, store
            score = evaluation(y_test, y_predict)
            result.append(score)
        
        # return as 2d array
        return np.asarray(result).reshape(n_iter, len(result)/n_iter)

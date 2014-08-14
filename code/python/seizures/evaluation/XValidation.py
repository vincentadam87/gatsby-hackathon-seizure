from sklearn.cross_validation import LeaveOneOut

import numpy as np
from seizures.evaluation.performance_measures import auc
from seizures.helper.data_structures import stack_matrices, stack_vectors


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
    def evaluate(X_list, y_list, predictor, evaluation=auc):
        """
        Performs LOO cross-validation on training data [X] and labels [y].
        Assumes that y is discrete.

        /// 
        Wittawat : If I understand the code below correctly, we are actually doing
        a k-fold cross validation where k is specified by the length of X_list. 
        This is not necessarily a LOO cross-validation. 
        ///
        
        Note that the training data/labels are in the form of a list of matrices,
        where each matrix contains feature vectors for one particular seizure.
        Cross-validation is done over those matrices, the classifier is then 
        trained on the concatenated matrices within those blocks.
        Labels are structured in the same way, list of vectors of labels for the
        corresponding matrices.
        
        Parameters:
        X_list     - training data, list of 2d numpy arrays
        y_list     - training labels, list of 1d numpy arrays, same length as above list
        predictor  - instance of PredictorBase
        evaluation - function handle that takes two equally sized 1d vectors
                     and evaluates some performance measure on them.
                     Optional, default is AUC
                     
        Returns:
        1d array where each entry corresponds to the performance measure the 
        test folds, same number as length of lists.
        
        @author: Heiko
        """
        # make sure we get right types
        assert(type(X_list) == type([]))
        assert(type(y_list) == type([]))
        assert(len(y_list) == len(X_list))
        for i in range(len(X_list)):
            assert(type(X_list[i]) == np.ndarray)
            assert(type(y_list[i]) == np.ndarray)
        
        # array sizes
        dim = X_list[0].shape[1]
        for i in range(len(X_list)):
            # make sure data in fold i is a matrix
            assert(len(X_list[i].shape) == 2)
            # make sure input and output have the same #instances 
            assert(len(y_list[i]) == X_list[i].shape[0])
            assert(X_list[i].shape[1] == dim)
            assert(len(y_list[i].shape) == 1)
        
        # create loo using sklearn
        # this is done on the list indices
        loo = LeaveOneOut(len(X_list))
        
        # run loo x-validation on the inner blocks
        result = []
        for train_index, test_index in loo:
            # partition data blocks, test data is matrix/list
            X_train_list = [X_list[i] for i in train_index]
            y_train_list = [y_list[i] for i in train_index]
            X_test = X_list[test_index[0]]
            y_test = y_list[test_index[0]]
            
            # concatenate matrices and vectors
            X_train = stack_matrices(X_train_list)
            y_train = stack_vectors(y_train_list)
            
            # make sure there is more than one class
            assert(len(np.unique(y_train)) > 1)
            assert(len(np.unique(y_test)) > 1)
            
            # fit model
            predictor.fit(X_train, y_train)
            
            # run on held out matrix
            y_predict = predictor.predict(X_test)
#            print y_predict
            
            # some sanity checks on the provided predictor to avoid problems
            if not type(y_predict) == np.ndarray:
                raise TypeError("Provided predictor doesn't return numpy array, but %s" % str(type(y_predict)))
            
            if not len(y_predict.shape) == 1:
                raise TypeError("Provided predictor doesn't not return 1d array")
            
            if not len(y_predict) == len(y_test):
                raise TypeError("Provided predictor doesn't return right number of labels")
            
            # evaluate, store
            score = evaluation(y_test, y_predict)
            result.append(score)
        
        # return as 2d array
        return np.asarray(result)

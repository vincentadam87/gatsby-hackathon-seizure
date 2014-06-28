from numpy.matlib import repmat

import numpy as np


class ToyXValidationData():
    """
    Class that generates easy data for XValidation (list of matrices/lists)
    to test the pipeline
    
    @author Heiko
    """
    @staticmethod
    def get():
        X_list = []
        y_list = []
        dim = 10
        for _ in range(10):
            n = np.random.randint(10, 100)
            y = (np.random.rand(n) > .5).astype(np.int64)
            X = np.random.randn(n, dim) + repmat(y, dim, 1).T
            X_list += [X]
            y_list += [y]
            
        return X_list, y_list

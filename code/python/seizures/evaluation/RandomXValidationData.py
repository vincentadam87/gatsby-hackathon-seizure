import numpy as np


class RandomXValidationData():
    """
    Class that generates random data for XValidation (list of matrices/lists)
    to test the pipeline
    
    @author Heiko
    """
    @staticmethod
    def get():
        X_list = []
        y_list = []
        dim = 100
        for _ in range(10):
            n = np.random.randint(100, 1000)
            X_list += [np.random.randn(n, dim)]
            y_list += [(np.random.rand(n)>.5).astype(np.int64)]
            
        return X_list, y_list

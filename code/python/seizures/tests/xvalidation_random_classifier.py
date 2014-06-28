'''
Created on 28 Jun 2014

@author: heiko
'''
import numpy as np
from seizures.evaluation.XValidation import XValidation
from seizures.prediction.RandomPredictor import RandomPredictor


if __name__ == '__main__':
    predictor = RandomPredictor()
    
    N = 1000
    D = 2
    X = np.random.randn(N, D)
    y = np.random.randint(0, 2, N)
    
    print XValidation.evaluate(X, y, predictor, n_iter=20, test_size=0.5)

'''
Created on 28 Jun 2014

@author: heiko
'''
import numpy as np
from seizures.evaluation.XValidation import XValidation
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor


def test_predictor(predictor_cls):
    predictor = predictor_cls()

    N=1000
    D=2
    X=np.random.randn(N,D)
    y=np.random.randint(0,2,N)

    print XValidation.evaluate(X, y, predictor, n_iter=2, test_size=0.5)

if __name__ == '__main__':
    print "RandomPredictor"
    test_predictor(RandomPredictor)

    print "ForestPredictor"
    test_predictor(ForestPredictor)
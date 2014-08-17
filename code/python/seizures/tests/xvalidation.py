'''
Created on 28 Jun 2014

@author: heiko
'''
import numpy as np
from seizures.evaluation.XValidation import XValidation
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.prediction.SVMPredictor import SVMPredictor


def test_predictor(predictor_cls):
    predictor = predictor_cls()

    N=1000
    D=2
    # simulate a 2-fold cross validation
    Xs = [np.random.randn(N/2, D), np.random.randn(N/2, D)]
    ys = [np.random.randint(0, 2, N/2), np.random.randint(0, 2, N/2)]
    #X=np.random.randn(N,D)
    #y=np.random.randint(0,2,N)

    print XValidation.evaluate(Xs, ys, predictor)

if __name__ == '__main__':
    print "RandomPredictor"
    test_predictor(RandomPredictor)

    print "ForestPredictor"
    test_predictor(ForestPredictor)

    print "SVMPredictor"
    test_predictor(SVMPredictor)

'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.evaluation.RandomXValidationData import RandomXValidationData
from seizures.evaluation.XValidation import XValidation
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.prediction.SVMPredictor import SVMPredictor


def test_predictor(predictor_cls):
    predictor = predictor_cls()

    X_list, y_list = RandomXValidationData.get()

    print XValidation.evaluate(X_list, y_list, predictor)

if __name__ == '__main__':
    print "RandomPredictor"
    test_predictor(RandomPredictor)
# 
#     print "ForestPredictor"
#     test_predictor(ForestPredictor)
# 
#     print "SVMPredictor"
#     test_predictor(SVMPredictor)
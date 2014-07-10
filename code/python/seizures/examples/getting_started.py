'''
Created on 28 Jun 2014
@author: heiko, vincent
'''

# Loading necessary packages
from seizures.data.DataLoader import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy
from seizures.features.ARFeatures import ARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor
import numpy as np


def test_predictor(predictor_cls):
    ''' function that loads data for Dog_1 run crossvalidation with ARFeatures 
        INPUT:
        - predictor_cls: a Predictor class (implement)  
    '''

    # instanciating a predictor object from Predictor class
    predictor = predictor_cls()    

    # path to data (here path from within gatsby network)
    data_path = "/nfs/data3/kaggle_seizure/"

    # creating instance of autoregressive features
    feature_extractor = ARFeatures()
    
    # loading the data
    loader = DataLoader(data_path, feature_extractor)
    X_list = loader.training_data("Dog_1")
    y_list = loader.labels("Dog_1")

    # running cross validation    
    conditioned = [a * b for (a, b) in zip(y_list[0], y_list[1])]
    print XValidation.evaluate(X_list, conditioned, predictor, evaluation=accuracy)

if __name__ == '__main__':
    # code run at script launch

    print "ForestPredictor"
    test_predictor(ForestPredictor)

    print "SVMPredictor"
    test_predictor(SVMPredictor)

 

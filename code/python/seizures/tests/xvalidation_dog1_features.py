'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.data.DataLoader import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy
from seizures.features.FFTFeatures import FFTFeatures
from seizures.helper.data_path import get_data_path
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.prediction.SVMPredictor import SVMPredictor

import numpy as np

def test_predictor(predictor_cls):
    predictor = predictor_cls()
    
    data_path = get_data_path("data_path.txt")

    # arbritary
    band_means = np.linspace(0, 200, 66)
    band_width = 2
    feature_extractor = FFTFeatures(band_means=band_means, band_width=band_width)
    
    loader = DataLoader(data_path, feature_extractor)
    X_list = loader.training_data("Dog_1")
    y_list = loader.labels("Dog_1")

    print XValidation.evaluate(X_list, y_list[0], predictor, evaluation=accuracy)
    print XValidation.evaluate(X_list, y_list[1], predictor, evaluation=accuracy)

if __name__ == '__main__':
    print "RandomPredictor"
    test_predictor(RandomPredictor)

    print "ForestPredictor"
    test_predictor(ForestPredictor)

    print "SVMPredictor"
    test_predictor(SVMPredictor)

 
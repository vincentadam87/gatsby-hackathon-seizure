'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.evaluation.XValidation import XValidation
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.data.DataLoader import DataLoader
from seizures.features.FFTFeatures import FFTFeatures


def test_predictor(predictor_cls):
    predictor = predictor_cls()

    feature_extractor = FFTFeatures()
    data_path = "/home/heiko/data/seizure/"
    loader = DataLoader(data_path, feature_extractor)
    X_list = loader.training_data("Dog_1")
    y_list = loader.labels("Dog_1")

    print XValidation.evaluate(X_list, y_list[0], predictor)
    print XValidation.evaluate(X_list, y_list[1], predictor)

if __name__ == '__main__':
    print "RandomPredictor"
    test_predictor(RandomPredictor)

    print "ForestPredictor"
    test_predictor(ForestPredictor)
 
    print "SVMPredictor"
    test_predictor(SVMPredictor)
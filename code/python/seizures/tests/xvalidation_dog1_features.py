'''
Created on 28 Jun 2014

@author: heiko
'''
from os.path import expanduser

from seizures.data.DataLoader import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy
from seizures.features.FFTFeatures import FFTFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.prediction.SVMPredictor import SVMPredictor


def test_predictor(predictor_cls):
    predictor = predictor_cls()
    
    home = expanduser("~")
    f = open(home + "/data_path.txt")
    data_path = f.readline()
    print "data_path", data_path
    f.close()

    feature_extractor = FFTFeatures()
    
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
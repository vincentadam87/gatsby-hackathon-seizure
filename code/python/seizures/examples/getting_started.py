'''
Created on 28 Jun 2014
@author: heiko, vincent
'''

# to run this script, you need to tell python where to find your code
# 1- $gedit ~/.bashrc
# 2- add to .bashrc the following line: export PYTHONPATH=$PYTHONPATH:path_to_repo/code/python, save, start a new terminal
# 3- run this script: $python getting_started.py

 


# Loading necessary packages
import numpy as np

import sys
# you can also manually declare path
path_to_repo = "/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/code/python/"
sys.path.insert(1,path_to_repo)

from seizures.data.DataLoader import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy
from seizures.features.ARFeatures import ARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.features.MixFeatures import MixFeatures


def test_predictor(predictor_cls):
    ''' function that loads data for Dog_1 run crossvalidation with ARFeatures 
        INPUT:
        - predictor_cls: a Predictor class (implement)  
    '''

    # instanciating a predictor object from Predictor class
    predictor = predictor_cls()    

    # path to data (here path from within gatsby network)
    data_path = "/nfs/data3/kaggle_seizure/scratch/Stiched_data/Dog_1/"
    
    # creating instance of autoregressive features
    #feature_extractor = ARFeatures()
    band_means = np.linspace(0, 200, 66)
    band_width = 2
    FFTFeatures_args = {'band_means':band_means, 'band_width':band_width}

    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}},
                                     {'name':"FFTFeatures",'args':FFTFeatures_args}])
    #feature_extractor = ARFeatures()




    # loading the data
    loader = DataLoader(data_path, feature_extractor)
    X_list = loader.training_data("Dog_1")
    y_list = loader.labels("Dog_1")

    print 'printing data'

    # running cross validation    
    conditioned = [a * b for (a, b) in zip(y_list[0], y_list[1])]
    print XValidation.evaluate(X_list, conditioned, predictor, evaluation=accuracy)

if __name__ == '__main__':
    # code run at script launch

    print "ForestPredictor"
    test_predictor(ForestPredictor)

    print "SVMPredictor"
    test_predictor(SVMPredictor)

 

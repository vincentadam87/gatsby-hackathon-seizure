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
# assuming that you have manually added the path to repository to PYTHONPATH
# you can also manually declare path
#path_to_repo = "/nfs/data3/balaji/research/gatsby-hackathon-seizure/code/python/"
# path_to_repo = "/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/code/python/"
# sys.path.insert(1,path_to_repo)

from seizures.data.DataLoader import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from seizures.features.ARFeatures import ARFeatures
from seizures.features.MixFeatures import MixFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.Global import Global


def test_predictor(predictor_cls, patient_name='Dog_1'):
    ''' function that loads data for Dog_1 run crossvalidation with ARFeatures 
        INPUT:
        - predictor_cls: a Predictor class (implement)  
    '''

    # instanciating a predictor object from Predictor class
    predictor = predictor_cls()    

    # path to data (here path from within gatsby network)
    data_path = Global.path_map('clips_folder')
    
    # creating instance of autoregressive features
    #feature_extractor = ARFeatures()
    band_means = np.linspace(0, 200, 66)
    band_width = 2
    FFTFeatures_args = {'band_means':band_means, 'band_width':band_width}

#    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}},
#                                     {'name':"FFTFeatures",'args':FFTFeatures_args}])
    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}}])
#    feature_extractor = MixFeatures([{'name':"FFTFeatures",'args':FFTFeatures_args}])
    #feature_extractor = ARFeatures()

    # loading the data
    loader = DataLoader(data_path, feature_extractor)
    print loader.base_dir

    print '\npatient = %s' % patient_name
    X_list = loader.training_data(patient_name)
    y_list = loader.labels(patient_name)

    # separating the label
    early_vs_not = y_list[1] #[a * b for (a, b) in zip(y_list[0], y_list[1])]
    seizure_vs_not = y_list[0]
    
    # running cross validation    
#    conditioned = [a * b for (a, b) in zip(y_list[0], y_list[1])]
    print "\ncross validation: seizures vs not"
    result = XValidation.evaluate(X_list, seizure_vs_not, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(result), np.std(result), result)

    print "\ncross validation: early_vs_not"
    result = XValidation.evaluate(X_list, early_vs_not, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
          % (np.mean(result), np.std(result), result)

    # generate prediction for test data

if __name__ == '__main__':
    # code run at script launch

    # patient_name is e.g., Dog_1
    patient_name = sys.argv[1]
    
    print "ForestPredictor"
    test_predictor(ForestPredictor, patient_name)

#    print "\nSVMPredictor"
#    test_predictor(SVMPredictor, patient_name)

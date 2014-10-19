'''
Created on 10 August 2014
@author: vincent
'''

# Loading necessary packages
import numpy as np
import sys


from seizures.data.DataLoader_v2 import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.MixFeatures import MixFeatures
from seizures.features.SEFeatures import SEFeatures

from seizures.features.StatsFeatures import StatsFeatures

from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.ARFeatures import ARFeatures
from seizures.features.LyapunovFeatures import LyapunovFeatures

from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor

from seizures.prediction.XtraTreesPredictor import XtraTreesPredictor

from seizures.Global import Global
from sklearn.cross_validation import train_test_split


def Xval_on_single_patient(predictor_cls, feature_extractor, patient_name="Dog_1",preprocess=True):
    """
    Single patient cross validation
    Returns 2 lists of cross validation performances
    :param predictor_cls:
    :param feature_extractor
    :param patient_name:
    :return:
    """
    # predictor_cls is a handle to an instance of PredictorBase
    # Instantiate the predictor 
    predictor = predictor_cls()
    base_dir = Global.path_map('clips_folder')
    base_dir = '/nfs/data3/kaggle_seizure/clips/'
    loader = DataLoader(base_dir, feature_extractor)

    X_list,y_seizure, y_early = loader.blocks_for_Xvalidation(patient_name,preprocess=preprocess)
    #X_train,y_seizure, y_early = loader.training_data(patient_name)
    #y_train = [y_seizure,y_early]
    #X_list,y_list = train_test_split(X_train,y_train)

    # running cross validation
    print patient_name
    print "\ncross validation: seizures vs not"
    result_seizure = XValidation.evaluate(X_list, y_seizure, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(result_seizure), np.std(result_seizure), result_seizure)
    print "\ncross validation: early_vs_not"
    result_early = XValidation.evaluate(X_list, y_early, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
          % (np.mean(result_early), np.std(result_early), result_early)
    return result_seizure,result_early


def Xval_on_patients(predictor_cls, feature_extractor, patients_list=['Dog_1'],preprocess=True):
    ''' Runs cross validation for given predictor class and feature instance on the given list of patients
        INPUT:
        - predictor_cls: a Predictor class (implement)
        - feature_extractor: an instanciation of a Features class
        - patients_list: a list of subject strings e.g., ['Dog_1', 'Patient_2']
    '''

    assert(isinstance(feature_extractor, FeatureExtractBase))
    results_seizure = []
    results_early = []
    for patient_name in patients_list:
        result_seizure, result_early = Xval_on_single_patient(predictor_cls, feature_extractor, patient_name, preprocess=preprocess)
        results_seizure.append(result_seizure)
        results_early.append(result_early)

    avg_results_seizure = np.mean(np.array(results_seizure),axis=0)
    avg_results_early = np.mean(np.array(results_early),axis=0)
    print "\ncross validation: seizures vs not (ACROSS ALL SUBJECTS)"
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(avg_results_seizure), np.std(avg_results_seizure), avg_results_seizure)
    print "\ncross validation: early_vs_not (ACROSS ALL SUBJECTS)"
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
          % (np.mean(avg_results_early), np.std(avg_results_early), avg_results_early)
    return avg_results_seizure, avg_results_early
    # generate prediction for test data



def main():
    # code run at script launch
    #patient_name = sys.argv[1]

    # There are Dog_[1-4] and Patient_[1-8]
    patients_list = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 9)]
    patients_list =  ["Dog_%d" % i for i in [1]]  #["Patient_%d" % i for i in range(1, 9)]#++
 
    #feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}}])
    #feature_extractor = PLVFeatures()
    #feature_extractor = MixFeatures([{'name':"PLVFeatures",'args':{}},{'name':"ARFeatures",'args':{}}])
    #feature_extractor = ARFeatures()
    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}},{'name':"PLVFeatures",'args':{}},{'name':'SEFeatures','args':{}}])
    #feature_extractor = SEFeatures()
    #feature_extractor = LyapunovFeatures()

    #feature_extractor = StatsFeatures()

    preprocess = True
    predictor = SVMPredictor
    #predictor = XtraTreesPredictor

    if preprocess==True:
        print 'Preprocessing ON'
    else:
        print 'Preprocessing OFF'

    print 'predictor: ',predictor
    Xval_on_patients(predictor,feature_extractor, patients_list,preprocess=preprocess)

if __name__ == '__main__':
    main()




'''
Created on 10 August 2014
@author: vincent
'''

# Loading necessary packages
import numpy as np
import sys

from seizures.preprocessing.PreprocessingLea import PreprocessingLea
from seizures.data.DataLoader_slurm import DataLoader_slurm
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.MixFeatures import MixFeatures, StackFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.features.FeatureSplitAndStack import FeatureSplitAndStack
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.ARFeatures import ARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.Global import Global


def Xval_on_single_patient(predictor_cls, feature_extractor, patient_name="Dog_1",preprocess=None,max_segments=None):
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
    loader = DataLoader_slurm(base_dir, feature_extractor, preprocess=preprocess)
    X_list, y_preictal = loader.blocks_for_Xvalidation(patient_name, n_fold=3, max_segments=max_segments)


    # running cross validation
    print patient_name
    print "\ncross validation: seizures vs not"
    result_seizure = XValidation.evaluate(X_list, y_preictal, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(result_seizure), np.std(result_seizure), result_seizure)

    return result_seizure


def Xval_on_patients(predictor_cls, feature_extractor, patients_list=['Dog_1'],preprocess=None,max_segments=None):
    ''' Runs cross validation for given predictor class and feature instance on the given list of patients
        INPUT:
        - predictor_cls: a Predictor class (implement)
        - feature_extractor: an instanciation of a Features class
        - patients_list: a list of subject strings e.g., ['Dog_1', 'Patient_2']
    '''

    assert(isinstance(feature_extractor, FeatureExtractBase))
    results_seizure = []

    for patient_name in patients_list:
        result_seizure = Xval_on_single_patient(predictor_cls, feature_extractor, patient_name, preprocess=preprocess,max_segments=max_segments)
        results_seizure.append(result_seizure)

    avg_results_seizure = np.mean(np.array(results_seizure), axis=0)
    print "\ncross validation: preictal vs interictal"
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(avg_results_seizure), np.std(avg_results_seizure), avg_results_seizure)

    return avg_results_seizure



def main():
    # code run at script launch

    # There are Dog_[1-5] and Patient_[1-2]
    patients_list = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 2)]
    patients_list =  ["Dog_1" ]

    feature1 = ARFeatures()
    feature2 = PLVFeatures()
    feature3 = SEFeatures()
    feature_extractor = StackFeatures(feature1, feature2, feature3)
    stack_feature =FeatureSplitAndStack(feature_extractor, 60)

    preprocess = PreprocessingLea()
    predictor = ForestPredictor

    if preprocess!=None:
        print 'Preprocessing ON'
    else:
        print 'Preprocessing OFF'

    print 'predictor: ', predictor
    Xval_on_patients(predictor,
                     stack_feature,
                     patients_list,
                     preprocess=preprocess,
                     max_segments=50)

if __name__ == '__main__':
    main()




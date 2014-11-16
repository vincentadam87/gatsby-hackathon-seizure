'''
Created on 10 August 2014
@author: vincent
'''

# Loading necessary packages
import numpy as np
import sys, os
import time
from prettytable import PrettyTable

from seizures.preprocessing.PreprocessingLea import PreprocessingLea
from seizures.data.DataLoader_slurm import DataLoader_slurm
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.MixFeatures import MixFeatures, StackFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.features.FeatureSplitAndStack import FeatureSplitAndStack
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.ARFeatures import ARFeatures, VarLagsARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor

from seizures.Global import Global


def Xval_on_single_patient(predictor,
                           feature_extractor,
                           patient_name="Dog_1",
                           preprocess=None,
                           max_segments=None,
                           path_dict=None,
                           n_fold=3):
    """
    Single patient cross validation
    Returns 2 lists of cross validation performances
    :param predictor_cls:
    :param feature_extractor
    :param patient_name:
    :return:
    """

    loader = DataLoader_slurm(path_dict, feature_extractor, preprocess=preprocess)
    X_list, y_preictal = loader.blocks_for_Xvalidation(patient_name, n_fold=n_fold, max_segments=max_segments)


    # running cross validation
    print patient_name
    print "\ncross validation: seizures vs not"
    result_seizure = XValidation.evaluate(X_list, y_preictal, predictor, evaluation=auc)
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(result_seizure), np.std(result_seizure), result_seizure)




    return result_seizure


def Xval_on_patients(predictor,
                     feature_extractor,
                     patients_list=['Dog_1'],
                     preprocess=None,
                     max_segments=None,
                     path_dict=None,
                     n_fold=3):
    ''' Runs cross validation for given predictor class and feature instance on the given list of patients
        INPUT:
        - predictor_cls: a Predictor class (implement)
        - feature_extractor: an instanciation of a Features class
        - patients_list: a list of subject strings e.g., ['Dog_1', 'Patient_2']
    '''

    assert(isinstance(feature_extractor, FeatureExtractBase))
    results_seizure = []

    result_str_ind = ""

    for patient_name in patients_list:
        result_seizure = Xval_on_single_patient(predictor,
                                                feature_extractor,
                                                patient_name,
                                                preprocess=preprocess,
                                                max_segments=max_segments,
                                                path_dict=path_dict,
                                                n_fold=n_fold)
        results_seizure.append(result_seizure)
        result_str_ind += "Patient: "+patient_name+"\n"
        result_str_ind += "Predictor: "+str(predictor)+"\n"
        x = PrettyTable(["Feature(s)", "AUC mean", "AUC std"])
        x.float_format = "1.4"
        x.align["AUC mean"] = "l"
        x.align["AUC std"] = "l"
        x.add_row([str(feature_extractor), np.mean(result_seizure), np.std(result_seizure)])
        result_str_ind += str(x)+"\n"


    avg_results_seizure = np.mean(np.array(results_seizure), axis=0)
    print "\ncross validation: preictal vs interictal"
    print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
           % (np.mean(avg_results_seizure), np.std(avg_results_seizure), avg_results_seizure)

    # ------- pretty tabling
    result_str = "working with dataset: "+  path_dict['clips_folder'] +"\n"
    result_str += "Xval, nfold = "+ str(n_fold)+"\n"
    result_str += "Preprocessing:\n"+ str(preprocess)+"\n"
    result_str += result_str_ind
    result_str += "Patients: "+str(patients_list)+"\n"
    result_str += "Predictor: "+str(predictor)+"\n"
    x = PrettyTable(["Feature(s)", "AUC mean", "AUC std"])
    x.float_format = "1.4"
    x.align["AUC mean"] = "l"
    x.align["AUC std"] = "l"
    x.add_row([str(feature_extractor), np.mean(avg_results_seizure), np.std(avg_results_seizure)])
    result_str += str(x)+"\n"

    # -------- writing txt file

    result_folder = path_dict['my_xval_folder']
    filename = 'xval_result_'+ time.strftime("%Y%m%d-%H%M%S")+".txt"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    f = open(result_folder + filename, "w")
    print "Storing results to", filename
    f.write(str(result_str))
    f.close()


    print str(result_str)

    return avg_results_seizure



def main():
    # code run at script launch



    # Loop over
    # - features
    #   - features alone
    #   - combinations
    # - parameters
    #   -
    # - splitting
    # - downsampling ?
    # - predictors


    # Easy first test
    # features alone (3 PLV, AR, SE)
    # split and stack in 10 or not (2) or split and average
    # preprocess fs = 400, 800 (2)
    # forest trees # estimators 100 - 500 (2)

    #----------- Declare paths
    path_dict = Global.custom_path_dict('vincent')
    path_dict['clips_folder'] = '/nfs/data3/kaggle_prediction'

    # There are Dog_[1-5] and Patient_[1-2]
    patients_list = ["Dog_%d" % i for i in range(1, 2)] + ["Patient_%d" % i for i in range(1, 2)]

    Features = [VarLagsARFeatures(lags=2),
                VarLagsARFeatures(lags=3),
                PLVFeatures(),
                SEFeatures(fmax=1000,nband=60),
                SEFeatures(fmax=500,nband=60)]

    for feature in Features:
        for n_split in [1,10, 60]:
            for targetrate in [400, 800, 1600]:
                for n_estimators in [100,500,1000]:
                    #----------- Declare features
                    stack_feature = FeatureSplitAndStack(feature, n_split)
                    #----------- Declare predictors
                    predictor = ForestPredictor(n_estimators=n_estimators)
                    #----------- Declare preprocessing
                    preprocess = PreprocessingLea(targetrate=targetrate)
                    #----------- running cross-validation
                    Xval_on_patients(predictor,
                                     stack_feature,
                                     patients_list,
                                     preprocess=preprocess,
                                     max_segments=None,
                                     path_dict=path_dict,
                                     n_fold=3)

if __name__ == '__main__':
    main()




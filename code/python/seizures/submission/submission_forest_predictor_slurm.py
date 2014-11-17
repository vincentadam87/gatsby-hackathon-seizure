'''
Created on 28 Jun 2014

@author: heiko
'''
import numpy as np
import sys, os
import time
from prettytable import PrettyTable

from seizures.preprocessing.PreprocessingLea import PreprocessingLea
from seizures.data.DataLoader_slurm import DataLoader_slurm
from seizures.data.DataLoader import DataLoader

from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.MixFeatures import MixFeatures, StackFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.features.FeatureSplitAndStack import FeatureSplitAndStack
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.ARFeatures import ARFeatures, VarLagsARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.XtraTreesPredictor import XtraTreesPredictor
from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.MixFeatures import MixFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.SVMPredictor import SVMPredictor

from seizures.preprocessing.PreprocessingLea import PreprocessingLea
from seizures.submission.SubmissionFile_slurm import SubmissionFile_slurm
from seizures.Global import Global


import numpy as np
import sys


if __name__ == '__main__':

    patients = ["Dog_%d" % i for i in range(1, 6)] + ["Patient_%d" % i for i in range(1, 3)]

    targetrate = 800
    preprocess = PreprocessingLea(targetrate=targetrate)
    predictor = XtraTreesPredictor(n_estimators=500)
    feature_extractor = StackFeatures(VarLagsARFeatures(lags=2), PLVFeatures(), SEFeatures(fmax=1000,nband=100) )

    path_dict = Global.custom_path_dict('wittawat')
    path_dict['clips_folder'] = '/nfs/data3/kaggle_prediction'
    data_path = path_dict['clips_folder']
    submission = SubmissionFile_slurm(path_dict, patients=patients)
    submission.generate_submission(predictor,
                                   feature_extractor,
                                   output_fname=patients,
                                   preprocess=preprocess)

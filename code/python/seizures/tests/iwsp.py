import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os,sys

sys.path.append("/nfs/nhome/live/vincenta/git/independent-jobs")

from seizures.Global import Global

from seizures.features.ARFeatures import *
from seizures.features.MixFeatures import StackFeatures
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.Boosting import AdaBoostTrees
from seizures.pipelines.FeaturePredictorTest import CVFeaturesPredictorsTester
from seizures.pipelines.FeaturePredictorTest import CachedCVFeaPredTester


feature_extractors = [
    ARFeatures(),
    VarLagsARFeatures(2),
    VarLagsARFeatures(3)]#,
'''
VarLagsARFeatures(4),
    VarLagsARFeatures(6),
    SEFeatures(),
    PLVFeatures(),
    StackFeatures(VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
    StackFeatures(VarLagsARFeatures(2), SEFeatures()),
    StackFeatures(ARFeatures(),VarLagsARFeatures(2), SEFeatures()),
    StackFeatures(ARFeatures(), VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
    StackFeatures(VarLagsARFeatures(2), VarLagsARFeatures(4), PLVFeatures(), SEFeatures()),
    StackFeatures(VarLagsARFeatures(2), VarLagsARFeatures(6), PLVFeatures(), SEFeatures()),
]
'''

predictors = [ ForestPredictor(n_estimators=100, max_features=0.2)]
patients = ['Dog_4','Dog_1']
#patient = 'Patient_2'
max_segments=100


# Instruction
# Set patients to 'All' to have a single big classifier (but not very interesting)
# Set clips_folder to Global.path_map('clips_folder')+'clips_notest/' to work with all the data  (test label for all)


clips_folder = Global.path_map('clips_folder')

tables = []
for patient in patients:
    tester = CachedCVFeaPredTester(feature_extractors, predictors, patient, data_path=clips_folder)
    #tester = CVFeaturesPredictorsTester(feature_extractors, predictors, patient)
    # randomly select subsamples of total segments (ictal + interictal)
    table = tester.test_combination(fold=3, max_segments=max_segments)
    tables.append(table)

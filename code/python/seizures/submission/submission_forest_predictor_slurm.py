'''
Created on 28 Jun 2014

@author: heiko
'''

from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.MixFeatures import MixFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.preprocessing.PreprocessingLea import PreprocessingLea
from seizures.submission.SubmissionFile_slurm import SubmissionFile_slurm
from seizures.Global import Global


import numpy as np
import sys


if __name__ == '__main__':

    patients = ["Dog_%d" % i for i in range(1, 6)] + ["Patient_%d" % i for i in range(1, 3)]
    feature_list = [{'name':"ARFeatures",'args':{}},{'name':"PLVFeatures",'args':{}},{'name':'SEFeatures','args':{}},{'name':"StatsFeatures",'args':{}}]

    print '---------'
    print 'patients: '
    print patients
    print 'feature_list: '
    print feature_list


    preprocess = PreprocessingLea()
    predictor_seizure = ForestPredictor()
    feature_extractor = MixFeatures(feature_list)

    data_path = Global.path_map('clips_folder')

    submission = SubmissionFile_slurm(data_path, patients=patients)
    submission.generate_submission(predictor_seizure,
                                   feature_extractor,
                                   output_fname=patients,
                                   preprocess=preprocess)

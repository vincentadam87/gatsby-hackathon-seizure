'''
Created on 18 August 2014

@author: Wittawat 
'''

from seizures.features.FFTFeatures import FFTFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.submission.SubmissionFile import SubmissionFile
from seizures.Global import Global
from seizures.features.ARFeatures import *
from seizures.features.MixFeatures import StackFeatures
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.XCHUHFeatures import XCHUHFeatures
import numpy as np
import sys


if __name__ == '__main__':
    assert(len(sys.argv)>1)
    output_fname = sys.argv[1]
    if len(sys.argv)>2:
	    patient = sys.argv[2]
    else:
        patient = None
    feature_gen = StackFeatures(ARFeatures(),  PLVFeatures(), XCHUHFeatures())

    print '---------'
    print 'argv1: ' + output_fname
    print 'argv2: ' + patient
    print 'argv3: '
    print feature_gen
    predictor_seizure = ForestPredictor(n_estimators=200, max_features=0.2)
    predictor_early = ForestPredictor(n_estimators=200, max_features=0.2)
    

    test_files = None       # for submission
    test_files = 'train'    # for local evaluation
    data_path = Global.path_map('clips_folder')
    
    submission = SubmissionFile(data_path,patients=patient)
    submission.generate_submission(predictor_seizure, predictor_early,
                            feature_extractor, test_filenames=test_files,output_fname = output_fname)





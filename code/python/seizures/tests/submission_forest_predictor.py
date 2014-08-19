'''
Created on 28 Jun 2014

@author: heiko
'''

from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.MixFeatures import MixFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.submission.SubmissionFile import SubmissionFile
from seizures.Global import Global
import numpy as np
import sys


if __name__ == '__main__':
    assert(len(sys.argv)>1)
    output_fname = sys.argv[1]
    if len(sys.argv)>2:
	    patients = sys.argv[2]
    else:
        patients = None
    if len(sys.argv)>3:
        feature_list = eval(sys.argv[3])
    else:
        feature_list = [{'name':"ARFeatures",'args':{}},{'name':"PLVFeatures",'args':{}},{'name':'SEFeatures','args':{}},{'name':"StatsFeatures",'args':{}}]
        #[{'name':"ARFeatures",'args':{}}]


    print '---------'
    print 'argv1: ' + output_fname
    print 'argv2: ' + patients
    print 'argv3: '
    print feature_list

    preprocess=True

    predictor_seizure = ForestPredictor()
    predictor_early = ForestPredictor()

    #examples of feature use
    #feature_extractor = ARFeatures()
    #feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}}])

    band_means = np.linspace(0, 200, 66)
    band_width = 2
    #FFTFeatures_args = {'band_means':band_means, 'band_width':band_width}
    feature_extractor = MixFeatures(feature_list)

    test_files = None       # for submission
    test_files = 'train'    # for local evaluation
    data_path = Global.path_map('clips_folder')
    
    submission = SubmissionFile(data_path,patients=patients)
    submission.generate_submission(predictor_seizure, predictor_early,
                            feature_extractor, test_filenames=test_files,output_fname = output_fname,preprocess=preprocess)

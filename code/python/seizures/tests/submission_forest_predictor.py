'''
Created on 28 Jun 2014

@author: heiko
'''

from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.MixFeatures import MixFeatures
from seizures.helper.data_path import get_data_path
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.submission.SubmissionFile import SubmissionFile
import numpy as np


if __name__ == '__main__':
    predictor_seizure = ForestPredictor()
    predictor_early = ForestPredictor()

    #feature_extractor = ARFeatures()
    band_means = np.linspace(0, 200, 66)
    band_width = 2
    FFTFeatures_args = {'band_means':band_means, 'band_width':band_width}

#    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}},
#                                     {'name':"FFTFeatures",'args':FFTFeatures_args}])

    feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}}])
#    feature_extractor = MixFeatures([{'name':"FFTFeatures",'args':FFTFeatures_args}])

#    feature_extractor = FFTFeatures()
    
#    test_files = ["Dog_1_test_segment_1.mat"]
    data_path = get_data_path()
    
    submission = SubmissionFile(data_path)
    submission.generate_submission(predictor_seizure, predictor_early,
                            feature_extractor)  #, test_filenames=test_files)

'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.features.RandomFeatures import RandomFeatures
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.submission.SubmissionFile import SubmissionFile
from seizures.Global import Global


if __name__ == '__main__':
    predictor = RandomPredictor()
    extractor = RandomFeatures()
    
    test_files = ["Dog_1_test_segment_1.mat"]
    data_path = Global.path_map('clips_folder')
    
    submission = SubmissionFile(data_path)
    predictor_seizure = predictor 
    predictor_early = predictor 
    submission.generate_submission(predictor_seizure, predictor_early,
                            extractor, test_filenames=test_files)

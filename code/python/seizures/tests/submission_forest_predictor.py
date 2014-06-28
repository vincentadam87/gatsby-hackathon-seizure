'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.features.FFTFeatures import FFTFeatures
from seizures.helper.data_path import get_data_path
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.submission.SubmissionFile import SubmissionFile


if __name__ == '__main__':
    predictor = ForestPredictor()
    extractor = FFTFeatures()
    
    test_files = ["Dog_1_test_segment_1.mat"]
    data_path = get_data_path("data_path.txt")
    
    submission = SubmissionFile(data_path)
    submission.generate_submission(predictor, predictor,
                            extractor, test_filenames=test_files)
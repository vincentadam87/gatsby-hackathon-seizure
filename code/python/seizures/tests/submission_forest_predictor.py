'''
Created on 28 Jun 2014

@author: heiko
'''
from os.path import expanduser

from seizures.features.FFTFeatures import FFTFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.submission.SubmissionFile import SubmissionFile


if __name__ == '__main__':
    predictor = ForestPredictor()
    extractor = FFTFeatures()
    
    test_files = ["Dog_1_test_segment_1.mat"]
    
    home = expanduser("~")
    f = open(home + "/data_path.txt")
    data_path = f.readline()
    print "data_path", data_path
    f.close()
    
    submission = SubmissionFile(data_path)
    submission.generate_submission(predictor, predictor,
                            extractor, test_filenames=test_files)
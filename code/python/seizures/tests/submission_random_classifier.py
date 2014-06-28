'''
Created on 28 Jun 2014

@author: heiko
'''
from seizures.features.RandomFeatures import RandomFeatures
from seizures.prediction.RandomPredictor import RandomPredictor
from seizures.submission.SubmissionFile import SubmissionFile


if __name__ == '__main__':
    predictor = RandomPredictor()
    extractor = RandomFeatures()
    
    submission = SubmissionFile()
    SubmissionFile.generate_submission(predictor, predictor,
                            extractor)
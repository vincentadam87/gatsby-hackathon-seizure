import os
from pandas import DataFrame, read_csv

from pandas.DataFrame import to_csv
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.prediction.PredictorBase import PredictorBase


class SubmissionFile():
    """
    Class to generate submission files
    
    @author Heiko
    """
    
    @staticmethod
    def get_filename_frame():
        """
        Returns a data-frame with all filenames of the sample submission file
        """
        me = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "sampleSubmission.csv"
        return read_csv(fname)["clip"]
        
    def generate_submission(self, predictor_seizure, predictor_early,
                            feature_extractor, output_fname="output.csv"):
        """
        Generates a submission file for a given pair of predictors, which is already
        trained (i.e. fit method was called). Loops over all filenames in
        
        Parameters:
        predictor_seizure - Instance of PredictorBase, fitted on seizure
        predictor_early   - Instance of PredictorBase, fitted on early
        feature_extractor - Instance of FeatureExtractBase, to extract test features
        output_fname      - Optional filename for result submission file
        """
        # make sure given objects are valid
        assert(isinstance(predictor_seizure, PredictorBase))
        assert(isinstance(predictor_early, PredictorBase))
        assert(isinstance(feature_extractor, FeatureExtractBase))
        
        # load filenames
        fnames = SubmissionFile.get_filename_frame()
        
        # predict on test data
        result = DataFrame(columns=('clip', 'seizure', 'early'))
        for fname in enumerate(fnames):
            print "Predicting on " + fname
            
            # extract data and predict
            X = feature_extractor.extract_test(fname)
            pred_seizure = predictor_seizure.predict(X)
            pred_early = predictor_seizure.predictor_early(X)
            result.append({'clip':fname, 'seizure':pred_seizure, 'early':pred_early})
        
        to_csv(output_fname, result)
        
if __name__ == "__main__":
    
    
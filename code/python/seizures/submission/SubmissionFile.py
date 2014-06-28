import os

from seizures.data.DataLoader import DataLoader
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.helper.data_structures import stack_matrices, stack_vectors
from seizures.prediction.PredictorBase import PredictorBase


class SubmissionFile():
    """
    Class to generate submission files
    
    @author Heiko
    """
    
    def __init__(self, data_path):
        """
        Constructor
        
        Parameters:
        data_path   - / terminated path of test data
        """
        self.data_path = data_path
        
        # generate dog and patient record names
        self.patients = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 9)]
        # self.patients = ["Dog_1"]
    
    @staticmethod
    def get_submission_filenames():
        """
        Returns a data-frame with all filenames of the sample submission file
        """
        me = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "sampleSubmission.csv"
        
        f = open(fname)
        lines = f.readlines()
        f.close()
        
        return [line.split(",")[0] for line in lines[1:]]
        
    def generate_submission(self, predictor_seizure, predictor_early,
                            feature_extractor, output_fname="output.csv",
                            test_filenames=None):
        """
        Generates a submission file for a given pair of predictors, which will
        be trained on all training data per patient/dog instance.
        
        Parameters:
        predictor_seizure - Instance of PredictorBase, fixed parameters
        predictor_early   - Instance of PredictorBase, fixed parameters
        feature_extractor - Instance of FeatureExtractBase, to extract test features
        output_fname      - Optional filename for result submission file
        test_filename     - Optional list of filenames to produce results on,
                            default is to use all
        """
        # make sure given objects are valid
        assert(isinstance(predictor_seizure, PredictorBase))
        assert(isinstance(predictor_early, PredictorBase))
        assert(isinstance(feature_extractor, FeatureExtractBase))
        
        if test_filenames is None:
            test_filenames = SubmissionFile.get_submission_filenames()
        
        # predict on test data, iterate over patients and dogs
        # and the in that over all test files
        result_lines = []

        data_loader = DataLoader(self.data_path, feature_extractor)

        for patient in self.patients:
            # load training data
            X_list = data_loader.training_data(patient)
            
            # skip if no files
            if len(X_list) == 0:
                continue
            
            print "Loaded data for " + patient
            
            y_seizure_list, y_early_list = data_loader.labels(patient)
            
            X = stack_matrices(X_list)
            y_seizure = stack_vectors(y_seizure_list)
            y_early = stack_vectors(y_early_list)
        
            # train both models
            print "Training seizure for " + patient
            predictor_seizure.fit(X, y_seizure)
            print "Training early for " + patient
            predictor_early.fit(X, y_early)
            
            # find out filenames that correspond to patient/dog
            fnames_patient = []
            for fname in test_filenames:
                if patient in fname:
                    fnames_patient + [fname]
            
            # now predict on all test points
            for fname in fnames_patient:
                print "Loading test data for " + fname
                eeg_data = None  # EEGData(fname)
                X = feature_extractor.extract(eeg_data)
                
                # predict
                print "Predicting seizure for " + fname
                pred_seizure = predictor_seizure.predict(X)
                
                print "Predicting seizure for " + fname
                pred_early = predictor_seizure.predictor_early(X)
                
                # store
                result_lines.append(",".join([fname, str(pred_seizure), str(pred_early)]))

        f = open(self.data_path + output_fname, "w")
        for line in result_lines:
            f.write(line)
        f.close()


import os
import numpy as np
from itertools import izip
from seizures.data.EEGData import EEGData
from seizures.data.DataLoader import DataLoader
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.helper.data_structures import stack_matrices, stack_vectors
from seizures.helper.data_structures import test_stack_matrices, test_stack_vectors
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
#        self.patients = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 9)]
#        self.patients = ["Patient_%d" % i for i in range(5, 9)]
#        self.patients = ["Dog_1"]
#        self.patients = ["Dog_2"]
        self.patients = ["Patient_1"]
#        self.patients = ["Patient_4"]
    
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
        
    @staticmethod
    def get_train_filenames():
        """
        Returns a list with names of the training data files
        """
        me = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "train_filenames.txt"
        f = open(fname)
        lines = f.readlines()
        f.close()
        return [line.rstrip('\n') for line in lines]
        
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
        elif test_filenames == 'train':
            test_filenames = SubmissionFile.get_train_filenames()
        
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
            
            y_seizure_list, y_early_list = data_loader.labels(patient)
            print "Loaded data for " + patient
#            print y_seizure_list
#            print y_early_list
            
            X_train = stack_matrices(X_list)
#            print X_list[0].shape, X_list[1].shape
            y_seizure = stack_vectors(y_seizure_list)
            y_early = stack_vectors(y_early_list)
            print X_train.shape
#            print X_train
#            print y_seizure
#            print y_early
#            print X.shape, y_seizure.shape, y_early.shape
#            test_stack_matrices(X, X_list)
#            test_stack_vectors(y_seizure, y_seizure_list)
#            test_stack_vectors(y_early, y_early_list)
        
            # train both models
            print
            print "Training seizure for " + patient
            predictor_seizure.fit(X_train, y_seizure)
            print "Training early for " + patient
            predictor_early.fit(X_train, y_early)

            pred_seizure = predictor_seizure.predict(X_train)
            pred_early = predictor_early.predict(X_train)
            print 'Results on training data'
            print 'seizure\tearly\tp(seizure)\tp(early)'
            for y_1, y_2, p_1, p_2 in izip(y_seizure, y_early, pred_seizure, pred_early):
                print '%d\t%d\t%.3f\t%.3f' % (y_1, y_2, p_1, p_2)

            # find out filenames that correspond to patient/dog
            fnames_patient = []
            for fname in test_filenames:
                if patient in fname:
                    fnames_patient += [fname]
            
            #print fnames_patient
           
            X_train_2_tmp = []
            # now predict on all test points
            for i_fname, fname in enumerate(fnames_patient):
                print "\nLoading test data for " + fname
                # eeg_data = None  # EEGData(fname)
                fname_full = '/nfs/data3/kaggle_seizure/clips/' + patient + '/' + fname
                print fname, fname_full
                eeg_data_tmp = EEGData(fname_full)
                eeg_data = eeg_data_tmp.get_instances()
                print eeg_data
                assert len(eeg_data) == 1
                eeg_data = eeg_data[0]
                print eeg_data
                X = feature_extractor.extract(eeg_data)
#                print X.shape
                
                # reshape since predictor expects matrix
                X = X.reshape(1, len(X))
                print X.shape
                X_train_2_tmp.append(X)

                
                # predict (one prediction only)
                print "Predicting seizure for " + fname
                pred_seizure = predictor_seizure.predict(X)[0]
#                print predictor_seizure.predict(X)
#                print pred_seizure
                
                print "Predicting early for " + fname
                pred_early = predictor_early.predict(X)[0]
#                print pred_early

                # store
                result_lines.append(",".join([fname, str(pred_seizure), str(pred_early)]))

            X_train_2 = np.vstack(X_train_2_tmp)
            X_train_mean = np.mean(X_train, 0)
            X_train_sd = np.std(X_train, 0)
            n_train = X_train.shape[0]
            X_train_2_mean = np.mean(X_train_2, 0)
            X_train_2_sd = np.std(X_train_2, 0)
            n_train_2 = X_train_2.shape[0]
            print n_train, n_train_2
            print X_train_mean
            print X_train_2_mean
            print X_train_sd
            print X_train_2_sd
            assert n_train == n_train_2
            assert X_train_mean == X_train_2_mean
            assert X_train_sd == X_train_2_sd
        

        print "Storing results to", self.data_path + output_fname
        f = open(self.data_path + output_fname, "w")
        for line in result_lines:
            f.write(line + '\n')
        f.close()


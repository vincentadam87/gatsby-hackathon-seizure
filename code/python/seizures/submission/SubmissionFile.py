import os
from itertools import izip
from seizures.data.DataLoader_v2 import DataLoader
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.prediction.PredictorBase import PredictorBase
from seizures.Global import Global

class SubmissionFile():
    """
    Class to generate submission files
    
    @author Heiko
    """
    
    def __init__(self, data_path,patients=None):
        """
        Constructor
        
        Parameters:
        data_path   - / terminated path. This is the base folder 
        containing e.g., Dog_1/, Dog_2/, ....
        patients - a list of patient names e.g., ['Dog_1', 'Patient_2', ...]
        """
        if not os.path.isdir(data_path):
            raise ValueError('%s is not a directory.'%data_path)

        self.data_path = Global.path_map('clips_folder')
        #self.data_path = '/nfs/data3/kaggle_seizure/clips/'

        if patients == None:
            self.patients = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 9)]
        else:
            self.patients = [patients] # will only work for single subject here...
#        self.patients = ["Patient_%d" % i for i in range(5, 9)]
#        self.patients = ["Dog_1"]
#        self.patients = ["Dog_2"]
#        self.patients = ["Patient_1"]
#        self.patients = ["Patient_4"]
    
    @staticmethod
    def get_submission_filenames():
        """
        Returns a data-frame with all filenames of the sample submission file
        """
        me = os.path.dirname(os.path.realpath(__file__))
        # data_dir = full path to data/ folder
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
                            test_filenames=None, preprocess=True):
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
        
        test_filenames = SubmissionFile.get_submission_filenames()
        train_filenames = SubmissionFile.get_train_filenames()
        
        all_result_lines = []

        for patient in self.patients:
            result_lines = []


            loader = DataLoader(self.data_path, feature_extractor)
            # X_train is n x d
            X_train,y_seizure, y_early = loader.training_data(patient,preprocess=preprocess)

            print X_train.shape
            print y_seizure.shape

            # train both models
            #print
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
            test_fnames_patient = []
            for fname in test_filenames:
                if patient in fname:
                    test_fnames_patient += [fname]

            # now predict on all test points
            loader = DataLoader(self.data_path, feature_extractor)
            # X_test: n x d matrix
            X_test = loader.test_data(patient,preprocess=preprocess)
            test_fnames_patient = loader.files

            for ifname in range(len(test_fnames_patient)):
                fname = test_fnames_patient[ifname]
                # X is one instance 
                X = X_test[ifname,:]
                # [0] to extract probability out of the ndarray
                pred_seizure = predictor_seizure.predict(X)[0]
                pred_early = predictor_early.predict(X)[0]
                name = fname.split("/")[-1]
                result_lines.append(",".join([name, str(pred_seizure), str(pred_early)]))


            csv_fname = patient + '_' + output_fname + '.csv'
            csv_path = Global.get_child_result_folder(csv_fname)
            print "Storing results to", csv_fname
            f = open(csv_path, "w")
            f.write("clip,seizure,early\n")
            for line in result_lines:
                f.write(line + '\n')
            f.close()

            all_result_lines.append(result_lines)


       
        

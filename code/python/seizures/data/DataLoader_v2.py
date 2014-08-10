import glob
from os.path import join
import os;
from seizures.data.EEGData import EEGData
import numpy as np
from seizures.features.FFTFeatures import FFTFeatures


class DataLoader(object):
    """
    Class to load data for a patient
    Loading from individual files (not the stitched data)
    @author Vincent (from Shaun original loader)
    """

    def __init__(self, base_dir, feature_extractor):
        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.' % base_dir)
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor
        self.patient_name = None
        self.type_labels = None
        self.early_labels = None
        self.features_train = None
        self.features_test = None
        self.files = None
        self.files_nopath = None

    def load_data(self, patient_name, type='training'):
        """
        Loads data for a patient and a type of data into class variables
        No output
        :param name: str
        """
        self.patient_name = patient_name
        self._reset_lists()
        files = self._get_files_for_patient(type)
        np.random.seed(0)
        I = np.random.permutation(len(files))
        self.files = [files[i] for i in I[0:200]]
        for filename in files:
            self._load_data_from_file(patient_name, filename)

    def training_data(self, patient_name):
        """
        returns features as a matrix of 1D np.ndarray
        returns classification vectors as 1D np.ndarrays
        :param patient_name:
        :return: feature matrix and labels
        """
        print "\nLoading train data for " + patient_name
        self.load_data(patient_name, type='training')
        X = self._merge_vectors_into_matrix(self.features_train)
        return X, np.array(self.type_labels), np.array(self.early_labels)

    def blocks_for_Xvalidation(self, patient_name,n_fold =10):
        """
        returns
        - a list of 2D ndarrays of features
        - a list of 1D ndarrays of type_labels
        - a list of 1D ndarrays of early_labels
        :param patient_name:
        :return: feature matrix and labels
        """
        X,y1,y2 = self.training_data(patient_name)
        n = len(y1)
        l = n/n_fold
        print n,l
        np.random.seed(0)
        I = np.random.permutation(n)
        Xp = [ X[I[i*l:(i+1)*l],:] for i in range(n_fold)]
        y1p = [ y1[I[i*l:(i+1)*l]] for i in range(n_fold)]
        y2p = [ y2[I[i*l:(i+1)*l]] for i in range(n_fold)]
        return Xp,y1p,y2p

    def test_data(self, patient_name):
        """
        returns features as a matrix of 1D np.ndarray
        :rtype : numpy 2D ndarray
        :param name: str
        """
        print "\nLoading test data for " + patient_name
        self.load_data(patient_name, type='test')
        return self._merge_vectors_into_matrix(self.features_test)


    def _reset_lists(self):
        self.episode_matrices = []
        self.type_labels = []
        self.early_labels = []

    def _get_files_for_patient(self, type="training"):
        assert (type in ["test", "training"])
        if type == "training":
            self.features_train = []
            files = glob.glob(join(self.base_dir, self.patient_name + '/*ictal*'))

        elif type == "test":
            self.features_test = []
            self.early_labels = []
            self.type_labels = []
            files = glob.glob(join(self.base_dir, self.patient_name + '/*test*'))
        # files_nopath = [os.path.basename(x) for x in files]
        #self.files_nopath = files_nopath
        return files


    def _load_data_from_file(self, patient, filename):
        if not filename.find('test') == -1:
            self._load_test_data_from_file(patient, filename)
        else:
            self._load_training_data_from_file(patient, filename)


    def _load_training_data_from_file(self, patient, filename):
        """
        Loading single file training data
        :param patient:
        :param filename:
        :return:
        """
        #print "\nLoading train data for " + patient + filename
        eeg_data_tmp = EEGData(filename)
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1
        eeg_data = eeg_data[0]
        if filename.find('interictal') > -1:
            y_seizure=0
            y_early=0
        elif eeg_data.latency < 15:
            y_seizure=1
            y_early=1
        else:
            y_seizure=1
            y_early=0
        x = self.feature_extractor.extract(eeg_data)
        self.features_train.append(np.hstack(x))
        self.type_labels.append(y_seizure)
        self.early_labels.append(y_early)

    def _load_test_data_from_file(self, patient, filename):
        """
        Loading single file test data
        :param patient:
        :param filename:
        :return:
        """
        assert ( filename.find('test'))
        print "\nLoading test data for " + patient + filename
        eeg_data_tmp = EEGData(filename)
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1
        eeg_data = eeg_data[0]
        x = self.feature_extractor.extract(eeg_data)
        self.features_test.append(np.hstack(x))


    def _get_feature_vector_from_instance(self, instance):
        return self.feature_extractor.extract(instance)

    def _merge_vectors_into_matrix(self, feature_vectors):
        n = len(feature_vectors)
        d = len(feature_vectors[0])
        matrix = np.zeros((n, d))
        for i, _ in enumerate(feature_vectors):
            matrix[i, :] = feature_vectors[i].T
        return matrix

    def labels(self, patient):
        assert (self.patient_name == patient)
        return self.type_labels, self.early_labels


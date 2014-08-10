import glob
from os.path import join
import os;
from seizures.data.EEGData import EEGData
import numpy as np
from seizures.features.FFTFeatures import FFTFeatures
from itertools import izip


class DataLoader(object):
    """
    Class to load data for a patient
    Loading from individual files (not the stitched data)
    @author Vincent (from Shaun original loader)
    """
    def __init__(self, base_dir, feature_extractor):
        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.'%base_dir)
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor
        self.patient_name = None
        self.type_labels = None
        self.early_labels = None
        self.features_train = None
        self.features_test = None
        self.files = None

    def load_data(self, patient_name, type='training'):
        """
        Loads data for a patient
        :param name: str
        """
        self.patient_name = patient_name
        self._reset_lists()
        files = self._get_files_for_patient(type)[1:10]
        self.files = files
        for filename in files:
            self._load_data_from_file(patient_name, filename)

    def training_data(self, patient_name):
        """
        returns training data for a vector
        :param patient_name:
        :return: feature matrix and labels
        """
        self.load_data(patient_name, type='training')
        X = self._merge_vectors_into_matrix(self.features_train)
        return X, self.type_labels, self.early_labels

    def test_data(self, patient_name):
        """
        Returns a matrix of all the features for a given patients,
        across all episodes.
        :rtype : numpy matrix
        :param name: str
        """
        self.load_data(patient_name, type='test')
        return self._merge_vectors_into_matrix(self.features_test)


    def _reset_lists(self):
        self.episode_matrices = []
        self.type_labels = []
        self.early_labels = []

    def _get_files_for_patient(self,type="training"):
        assert(type in ["test", "training"])
        if type == "training":
            self.features_train = []
            files = glob.glob(join(self.base_dir, self.patient_name + '/*ictal*'))
        elif type == "test":
            self.features_test = []
            self.early_labels = []
            self.type_labels = []
            files = glob.glob(join(self.base_dir, self.patient_name + '/*test*'))
        return files


    def _load_data_from_file(self, patient, filename):
        if not filename.find('test')==-1:
            self._load_test_data_from_file(patient, filename)
        else:
            self._load_training_data_from_file(patient, filename)


    def _load_training_data_from_file(self, patient, filename):
            """
            Loading single file data
            :param patient:
            :param filename:
            :return:
            """
            y_seizure = []
            y_early = []
            print "\nLoading train data for " + patient + filename
            eeg_data_tmp = EEGData(filename)
            eeg_data = eeg_data_tmp.get_instances()
            assert len(eeg_data) == 1
            eeg_data = eeg_data[0]
            if filename.find('interictal') > -1:
                y_seizure.append(0)
                y_early.append(0)
            elif eeg_data.latency < 15:
                y_seizure.append(1)
                y_early.append(1)
            else:
                y_seizure.append(1)
                y_early.append(0)
            x = self.feature_extractor.extract(eeg_data)
            self.features_train.append(np.hstack(x))
            self.type_labels.append(y_seizure)
            self.early_labels.append(y_early)

    def _load_test_data_from_file(self, patient, filename):
            """
            Loading single file data
            :param patient:
            :param filename:
            :return:
            """
            assert( filename.find('test'))
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
        assert(self.patient_name == patient)
        return self.type_labels, self.early_labels

if __name__ == '__main__':
    base_dir = "/Users/Shaun/dev/gatsby-hackathon/data/Hackaton_seizure_data/"
    extractor = FFTFeatures()

    loader = DataLoader(base_dir, extractor)
    matrix = loader.training_data("Dog_1")

    print matrix
    print loader.labels("Dog_1")

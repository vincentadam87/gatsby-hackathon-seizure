import glob
from os.path import join
from seizures.data.EEGData import EEGData
import numpy as np
from seizures.features.FFTFeatures import FFTFeatures


class DataLoader(object):
    """
    Class to load all the training data for a patient.

    @author Shaun
    """
    def __init__(self, base_dir, feature_extractor):
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor

        self.patient = None
        self.episode_matrices = None
        self.type_labels = None
        self.early_labels = None

    def training_data(self, name):
        """
        Returns a matrix of all the features for a given patients,
        across all episodes.

        :rtype : numpy matrix
        :param name: str
        """
        self.patient = name
        self._reset_lists()

        files = self._get_files_for_patient()

        for filename in files:
            self._load_data_from_file(filename)

        return self.episode_matrices

    def _reset_lists(self):
        self.episode_matrices = []
        self.type_labels = []
        self.early_labels = []

    def _get_files_for_patient(self):
        files = glob.glob(join(self.base_dir, self.patient + '*'))
        return files

    def _load_data_from_file(self, filename):
        eeg = EEGData(filename)
        instances = eeg.get_instances()

        feature_vectors = []

        for instance in instances:
            new_features = self._get_feature_vector_from_instance(instance)
            feature_vectors.append(new_features)

        self.episode_matrices.append(self._merge_vectors_into_matrix(feature_vectors))
        self.type_labels.append(np.ones(len(feature_vectors), dtype=np.int8) * eeg.label)

        # Only have 1 if there is a seizure and it is early
        self.early_labels.append(np.array(map(self._is_early, instances), dtype=np.int8) * eeg.label)

    def _get_feature_vector_from_instance(self, instance):
        return self.feature_extractor.extract(instance)

    def _merge_vectors_into_matrix(self, feature_vectors):
        n = len(feature_vectors)
        d = len(feature_vectors[0])

        matrix = np.zeros((n, d))
        for i, _ in enumerate(feature_vectors):
            matrix[i, :] = feature_vectors[i].T

        return matrix

    @staticmethod
    def _is_early(instance):
        return 1 if instance.latency < 15 else 0

    def labels(self, patient):
        assert(self.patient == patient)
        return self.type_labels, self.early_labels

if __name__ == '__main__':
    base_dir = "/Users/Shaun/dev/gatsby-hackathon/data/Hackaton_seizure_data/"
    extractor = FFTFeatures()

    loader = DataLoader(base_dir, extractor)
    matrix = loader.training_data("Dog_1")

    print matrix
    print loader.labels("Dog_1")
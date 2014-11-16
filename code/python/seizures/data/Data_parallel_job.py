import numpy as np
import os

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.tools.Log import logger
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.EEGData import EEGData

# Define our custom Job, which inherits from base class IndependentJob

class Data_parallel_job(IndependentJob):
    def __init__(self, aggregator, base_dir, feature_extractor, patient_name, filename, preprocess=None):
        IndependentJob.__init__(self, aggregator)

        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.' % base_dir)
        if not isinstance(feature_extractor, FeatureExtractBase):
            raise ValueError('feature_extractor must be an instance of FeatureExtractBase')

        self.patient_name = patient_name
        self.filename = filename
        self.base_dir = base_dir
        self.feature_extractor = feature_extractor
        # patient_name = e.g., Dog_1
        self.patient_name = None
        # type_labels = a list of {0, 1}. Indicators of a preictal (1 for preictal).
        self.type_labels = None
        self.features_train = None
        # a list of numpy arrays. Each element is from feature_extractor.extract()
        self.features_test = None
        # list of file names in base_dir directory
        self.files = None
        self.files_nopath = None
        self.preprocess = preprocess

    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        logger.info("computing")
        # create ScalarResult instance
        x, y = self._load_data_from_file(self.patient_name, self.filename)
        result = SingleResult([x,y])

        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done computing")


    def _load_data_from_file(self, patient, filename):
        if filename.find('test') != -1:
            # if filename is a test segment
            return self._load_test_data_from_file(patient, filename)
        else:
            return self._load_training_data_from_file(patient, filename)


    def _load_training_data_from_file(self, patient, filename):
        """
        Loading single file training data
        :param patient:
        :param filename:
        :return:
        """
        #print "\nLoading train data for " + patient + filename
        eeg_data_tmp = EEGData(filename)
        # a list of Instance's
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1
        # eeg_data is now an Instance
        eeg_data = eeg_data[0]
        if filename.find('interictal') > -1:
            y_interictal = 0
        else:
            y_interictal = 1
        fs = eeg_data.sampling_rate
        # preprocessing
        data = eeg_data.eeg_data
        ### comment if no preprocessing
        if self.preprocess!=None:
            eeg_data.eeg_data = self.preprocess.apply(data, fs)
        ###
        x = self.feature_extractor.extract(eeg_data)

        return np.hstack(x), y_interictal

    def _load_test_data_from_file(self, patient, filename):
        """
        Loading single file test data
        :param patient:
        :param filename:
        :return:
        """
        assert ( filename.find('test'))
        #print "\nLoading test data for " + patient + filename
        eeg_data_tmp = EEGData(filename)
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1
        eeg_data = eeg_data[0]

        fs = eeg_data.sample_rate
        # preprocessing
        data = eeg_data.eeg_data

        #params = self.params
        #params['fs']=fs

        ### comment if no preprocessing
        if self.preprocess!=None:
            eeg_data.eeg_data = self.preprocess.apply(data, fs)
        x = self.feature_extractor.extract(eeg_data)
        #self.features_test.append(np.hstack(x))
        return np.hstack(x)









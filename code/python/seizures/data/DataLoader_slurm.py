import glob
from os.path import join
import os
from seizures.data.EEGData import EEGData
import numpy as np
import random
from seizures.features.FeatureExtractBase import FeatureExtractBase
from os.path import expanduser

import independent_jobs
from independent_jobs.tools.Log import Log
from independent_jobs.tools.Log import logger
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from seizures.data.Data_parallel_job import Data_parallel_job

# Wittawat: Many load_* methods do not actually use the patient_name argument

class DataLoader_slurm(object):
    """
    Class to load data (all segments) for a patient
    Loading from individual files (not the stitched data)
    @author Vincent (from Shaun original loader)
    """

    def __init__(self, base_dir, feature_extractor, preprocess=None):
        """
        base_dir: path to the directory containing patient folders i.e., directory
        containing Dog_1/, Dog_2, ..., Patient_1, Patient_2, ....
        """
        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.' % base_dir)
        if not isinstance(feature_extractor, FeatureExtractBase):
            raise ValueError('feature_extractor must be an instance of FeatureExtractBase')

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

    def load_data(self, patient_name, type='training', max_segments=None):
        """
        Loads data for a patient and a type of data into class variables
        No output
        :param patient_name: e.g., Dog_1
        :param type: training or test
        :param max_segments: maximum segments to load. -1 to use the number of
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement.
        """
        self.patient_name = patient_name
        self._reset_lists()
        # For type='training', this will get all interictal and ictal file names
        files = self._get_files_for_patient(type)

        # reorder files so as to mix a bit interictal and ictal (to fasten
        # debug I crop the files to its early entries)
        print 'Load with extractor = %s'% (str(self.feature_extractor))
        #print 'Load with params = %s'% (str(self.params))

        random.seed(0)
        if type == 'training':
            print 'files: ', files
            files_interictal = [f for f in files if f.find("interictal_") >= 0 ]
            random.shuffle(files_interictal)
            print '%d interictal segments for %s'%(len(files_interictal), patient_name)
            files_preictal = [f for f in files if f.find("preictal_") >= 0]
            random.shuffle(files_preictal)
            print '%d preictal segments for %s'%(len(files_preictal), patient_name)

            files = []
            # The following loop just interleaves preictal and interictal segments
            # so that we have
            #[preictal_segment1, interictal_segment1, preictal_segment2, ...]
            for i in range(max(len(files_preictal), len(files_interictal))):
                if i < len(files_preictal):
                    files.append(files_preictal[i])
                if i < len(files_interictal):
                    files.append(files_interictal[i])

        #np.random.seed(0)
        #I = np.random.permutation(len(files))
        #I = range(len(files))

        if type == 'training':
            total_segments = len(files_interictal) + len(files_preictal)
        else:
            total_segments = len(files)

        if max_segments is None:
            subsegments = total_segments
        else:
            subsegments = min(max_segments, total_segments)

        print 'subsampling from %d segments to %d'% (total_segments, subsegments)
        #self.files = [files[i] for i in I[0:subsegments]]
        self.files = files[0:subsegments]

        #self.files = [files[i] for i in I[0:200]]

        if type == 'test':
            self.files = files


        #-------------------------


        Log.set_loglevel(20)
        logger.info("Start")
        # create an instance of the SGE engine, with certain parameters

        # create folder name string
        home = expanduser("~")
        foldername = os.sep.join([home, "kaggle"])
        logger.info("Setting engine folder to %s" % foldername)

        # create parameter instance that is needed for any batch computation engine
        logger.info("Creating batch parameter instance")
        batch_parameters = BatchClusterParameters(foldername=foldername)

        # create slurm engine (which works locally)
        logger.info("Creating slurm engine instance")
        engine = SlurmComputationEngine(batch_parameters)

        # we have to collect aggregators somehow
        aggregators = []

        # submit job n times
        logger.info("Starting loop over job submission")

        for i, filename in enumerate(self.files):
            logger.info("Submitting job %d" % i)
            print filename
            #print float(i)/float(len(self.files))*100.," percent complete         \r",
            # Each call of _load_data_from_file appends data to features_train
            # features_test lists depending on the (type) variable.
            # It also appends data to type_labels and early_labels.
            job = Data_parallel_job(SingleResultAggregator(), self.base_dir,
                                    self.feature_extractor,
                                    patient_name,
                                    filename,
                                    preprocess=self.preprocess)

            aggregators.append(engine.submit_job(job))

        # let the engine finish its business
        logger.info("Wait for all call in engine")
        engine.wait_for_all()

        # the reduce part
        # lets collect the results
        results = np.zeros(len(aggregators))


        logger.info("Collecting results")
        for i in range(len(aggregators)):
            logger.info("Collecting result %d" % i)
            # let the aggregator finalize things, not really needed here but in general
            aggregators[i].finalize()

            # aggregators[i].get_final_result() returns a ScalarResult instance,
            # which we need to extract the number from

            xy_list = aggregators[i].get_final_result().result

            self.features_train.append(xy_list[0])
            self.type_labels.append(xy_list[1])


        print "\ndone"

        # do the wait

        # agregate

        # push to self



    def training_data(self, patient_name, max_segments=None):
        """
        returns features as a matrix of 1D np.ndarray
        returns classification vectors as 1D np.ndarrays
        :param patient_name:
        :param max_segments: maximum segments to load. -1 to use the number of
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement.
        :return: feature matrix and labels
        """
        print "\nLoading train data for " + patient_name
        self.load_data(patient_name, type='training', max_segments=max_segments)
        X = self._merge_vectors_into_matrix(self.features_train)
        return X, np.array(self.type_labels)

    def blocks_for_Xvalidation(self, patient_name, n_fold=3, max_segments=None):
        """
        Stratified partitions (partition such that class proportion remains same
        in each data fold) of data for cross validation. The sum of instances
        in all partitions may be less than the original total.

        :param max_segments: maximum segments to load. -1 to use the number of
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement.

        returns
        - a list of 2D ndarrays of features
        - a list of 1D ndarrays of type_labels
        All the lists have length = n_fold.
        These outputs can be used with XValidation.

        :param patient_name:
        :return: feature matrix and labels
        """
        # y1 = type_labels,  y2 = early_labels
        X,y1 = self.training_data(patient_name, max_segments=max_segments)
        n = len(y1)
        assert(np.sum(y1)>n_fold)
        # do blocks for labels=1 and labels=0 separetaly and merge at the end
        # the aim is to always have both labels in each train/test sets
        Iy_1 = np.where([(y1[i]==1) for i in range(len(y1))])[0].tolist()
        Iy_0 = np.where([(y1[i]==0) for i in range(len(y1))])[0].tolist()

        def chunks(l, n_block):
            """ Yield n_block blocks.
            """
            m = len(l)/n_block
            r = len(l)%n_block
            for i in xrange(0, n_block):
                if i == n_block-1:
                    yield l[i*m:-1]
                else:
                    yield l[i*m:(i+1)*m]

        Iy_1_list = list(chunks(Iy_1, n_fold))
        Iy_0_list = list(chunks(Iy_0, n_fold))

        assert(len(Iy_1_list)==n_fold)

        Iy = [Iy_1_list[i]+Iy_0_list[i] for i in range(n_fold)]

        y1p = [ y1[Iy[i]] for i in range(n_fold)]
        Xp = [ X[Iy[i],:] for i in range(n_fold)]

        return Xp,y1p

    def test_data(self, patient_name, max_segments=None):
        """
        returns features as a matrix of 1D np.ndarray
        :rtype : numpy 2D ndarray
        :param name: str
        :param max_segments: maximum segments to load. -1 to use the number of
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement.
        """
        print "\nLoading test data for " + patient_name
        self.load_data(patient_name, type='test', max_segments=max_segments)
        return self._merge_vectors_into_matrix(self.features_test)


    def _reset_lists(self):
        self.episode_matrices = []
        self.type_labels = []

    def _get_files_for_patient(self, type="training"):
        assert (type in ["test", "training"])
        if type == "training":
            self.features_train = []
            print self.base_dir, '....', self.patient_name
            files = glob.glob(join(self.base_dir, self.patient_name + '/*ictal*'))
            #print 'fff: ', files

        elif type == "test":
            self.features_test = []
            self.type_labels = []
            files = glob.glob(join(self.base_dir, self.patient_name + '/*test*'))
        # files_nopath = [os.path.basename(x) for x in files]
        #self.files_nopath = files_nopath
        return files

    def _merge_vectors_into_matrix(self, feature_vectors):
        print feature_vectors
        n = len(feature_vectors)
        d = len(feature_vectors[0])
        matrix = np.zeros((n, d))
        for i, _ in enumerate(feature_vectors):
            matrix[i, :] = feature_vectors[i].T
        return matrix

    def labels(self, patient):
        # Wittawat: Why do we need patient argument ?
        assert (self.patient_name == patient)
        return self.type_labels


    def _get_feature_vector_from_instance(self, instance):
        return self.feature_extractor.extract(instance)


    # def _load_data_from_file(self, patient, filename):
    #     if filename.find('test') != -1:
    #         # if filename is a test segment
    #         self._load_test_data_from_file(patient, filename)
    #     else:
    #         self._load_training_data_from_file(patient, filename)
    #
    #
    # def _load_training_data_from_file(self, patient, filename):
    #     """
    #     Loading single file training data
    #     :param patient:
    #     :param filename:
    #     :return:
    #     """
    #     #print "\nLoading train data for " + patient + filename
    #     eeg_data_tmp = EEGData(filename)
    #     # a list of Instance's
    #     eeg_data = eeg_data_tmp.get_instances()
    #     assert len(eeg_data) == 1
    #     # eeg_data is now an Instance
    #
    #     eeg_data = eeg_data[0]
    #     if filename.find('interictal') > -1:
    #         y_interictal = 1
    #     else:
    #         y_interictal = 0
    #
    #     fs = eeg_data.sampling_rate
    #
    #     # preprocessing
    #     data = eeg_data.eeg_data
    #
    #     #params = self.params
    #     #params['fs']=fs
    #     ### comment if no preprocessing
    #     if self.preprocess!=None:
    #         eeg_data.eeg_data = self.preprocess.apply(data, fs)
    #     ###
    #     x = self.feature_extractor.extract(eeg_data)
    #     #print 'x: ', x.shape
    #     #self.features_train.append(np.hstack(x))
    #     #self.type_labels.append(y_interictal)
    #     return np.hstack(x), y_interictal
    #
    # def _load_test_data_from_file(self, patient, filename):
    #     """
    #     Loading single file test data
    #     :param patient:
    #     :param filename:
    #     :return:
    #     """
    #     assert ( filename.find('test'))
    #     print "\nLoading test data for " + patient + filename
    #     eeg_data_tmp = EEGData(filename)
    #     eeg_data = eeg_data_tmp.get_instances()
    #     assert len(eeg_data) == 1
    #     eeg_data = eeg_data[0]
    #
    #     fs = eeg_data.sample_rate
    #     # preprocessing
    #     data = eeg_data.eeg_data
    #
    #     #params = self.params
    #     #params['fs']=fs
    #
    #     ### comment if no preprocessing
    #     if self.preprocess!=None:
    #         eeg_data.eeg_data = self.preprocess.apply(data, fs)
    #     x = self.feature_extractor.extract(eeg_data)
    #     self.features_test.append(np.hstack(x))
    #
    #
    # def _get_feature_vector_from_instance(self, instance):
    #     return self.feature_extractor.extract(instance)
    #
    # def _merge_vectors_into_matrix(self, feature_vectors):
    #     print feature_vectors
    #     n = len(feature_vectors)
    #     d = len(feature_vectors[0])
    #     matrix = np.zeros((n, d))
    #     for i, _ in enumerate(feature_vectors):
    #         matrix[i, :] = feature_vectors[i].T
    #     return matrix
    #
    # def labels(self, patient):
    #     # Wittawat: Why do we need patient argument ?
    #     assert (self.patient_name == patient)
    #     return self.type_labels
    #

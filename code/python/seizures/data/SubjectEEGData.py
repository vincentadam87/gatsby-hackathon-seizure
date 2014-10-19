import glob
import random
from os.path import join
import os
from seizures.data.EEGData import EEGData
from seizures.preprocessing import preprocessing
import numpy as np
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.Global import Global


class SubjectEEGData(object):
    """
    Class to load data (all segments) for a patient
    Loading from individual files (not the stitched data)
    This is totally based on Vincent's DataLoader_v2. But unlike DataLoader, 
    This class separates out feature extraction step, allowing the loaded 
    data to be cached in case there are multiple feature extractors to try.
    Objects of this class may be treated as immutable.

    @author Wittawat 
    """

    def __init__(self, patient_name, base_dir=Global.path_map('clips_folder'),
            use_cache=True, max_train_segments=-1, max_test_segments=-1):
        """
        patient_name: for example, Dog_1
        base_dir: path to the directory containing patient folders i.e., directory 
        containing Dog_1/, Dog_2, ..., Patient_1, Patient_2, ....
        use_cache: if True, the loaded data is retained in memory so that the 
        call to data loading method will return immediately next time. 
        Require a large amount of memory. 
        max_XX_segments: maximum segments to load. -1 to use the number of 
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement. 

        """
        if not os.path.isdir(base_dir):
            raise ValueError('%s is not a directory.' % base_dir)

        # The followings attributes, once assigned, will not change.
        self.base_dir = base_dir
        # patient_name = e.g., Dog_1
        self.patient_name = patient_name
        self.use_cache = use_cache
        self.max_train_segments = max_train_segments
        self.max_test_segments = max_test_segments

        # this will be cached when get_train_data() is called. a list of training 
        # file names
        self.loaded_train_fnames = None
        # a list of loaded (Instance, y_seizure, y_early)
        self.loaded_train_data = None 
        # this will be cached when get_test_data() is called. a list of test  
        # file names
        self.loaded_test_fnames = None
        self.loaded_test_data = None

        # type_labels = a list of {0, 1}. Indicators of a seizure (1 for seizure).
        self.type_labels = None
        # early_labels = a list of {0, 1}. Indicators of an early seizure 
        # (1 for an early seizure).
        self.early_labels = None
        self.params = { 'anti_alias_cutoff': 500.,
            'anti_alias_width': 30.,
            'anti_alias_attenuation' : 40,
            'elec_noise_width' :3.,
            'elec_noise_attenuation' : 60.0,
            'elec_noise_cutoff' : [59.,61.],
            'targetrate':500}
    def get_train_data(self):
        """
        Loads training data for the patient data. 
        Return: a list of (Instance, y_seizure, y_early)'s
        """
        if self.loaded_train_data is None or not self.use_cache: 
            patient_train_file_list = self.get_patient_train_file_list()
            patient_name = self.patient_name
            # reorder files so as to mix a bit interictal and ictal (to fasten
            # debug I crop the files to its early entries)
            random.seed(0)
            files_interictal = [f for f in patient_train_file_list if f.find("_interictal_") >= 0 ]
            files_ictal = [f for f in patient_train_file_list if f.find("_ictal_") >= 0]
            print '%d interictal segments for %s'%(len(files_interictal), patient_name)
            print '%d ictal segments for %s'%(len(files_ictal), patient_name)

            # randomly shuffle lists 
            random.shuffle(files_interictal)
            random.shuffle(files_ictal)

            patient_train_file_list = []
            # The following loop just interleaves ictal and interictal segments
            # so that we have 
            #[ictal_segment1, interictal_segment1, ictal_segment2, ...]
            for i in range(max(len(files_ictal), len(files_interictal))):
                if i < len(files_ictal):
                    patient_train_file_list.append(files_ictal[i])
                if i < len(files_interictal):
                    patient_train_file_list.append(files_interictal[i])

            total_segments = len(files_interictal) + len(files_ictal)
            subsegments = min(self.max_train_segments, total_segments)
            print 'subsampling from %d segments to %d'% (total_segments, subsegments)
            loaded_train_fnames = patient_train_file_list[0:subsegments]

            train_data = []
            for i, filename in enumerate(loaded_train_fnames):
                print float(i)/len(loaded_train_fnames)*100.," percent complete         \r",
                # y_seizure, y_early are binary
                tr_instance, y_seizure, y_early = SubjectEEGData.load_train_data_from_file(patient_name, filename,self.params)
                train_data.append( (tr_instance, y_seizure, y_early) )
            print "\ndone"
            loaded_train_data = train_data

            if self.use_cache:
                # cached loaded data
                self.loaded_train_fnames = loaded_train_fnames
                self.loaded_train_data = loaded_train_data
        return self.loaded_train_data
       
    def get_test_data(self):
        """
        Loads test data for the patient 
        return a list of Instance's
        """

        if self.loaded_test_data is None or not self.use_cache: 
            patient_test_file_list = self.get_patient_test_file_list()
            patient_name = self.patient_name
            random.seed(0)
            patient_test_file_list = []

            total_segments = len(patient_test_file_list)
            subsegments = min(self.max_test_segments, total_segments)
            print 'subsampling from %d segments to %d'% (total_segments, subsegments)
            loaded_test_fnames = patient_test_file_list[0:subsegments]

            test_data = []
            for i, filename in enumerate(loaded_test_fnames):
                print float(i)/len(patient_test_file_list)*100.," percent complete         \r",
                te_instance = SubjectEEGData.load_test_data_from_file(patient_name, filename,self.params)
                test_data.append(te_instance)
            print "\ndone"
            loaded_test_data = test_data

            if self.use_cache:
                # cached loaded data
                self.loaded_test_fnames = loaded_test_fnames
                self.loaded_test_data = loaded_test_data
        return self.loaded_test_data


    def get_patient_test_file_list(self):
        return  glob.glob(join(self.base_dir, self.patient_name + '/*test*'))

    def get_patient_train_file_list(self):
        # These include interictal
        return  glob.glob(join(self.base_dir, self.patient_name + '/*ictal*'))

    #### static methods  ###
    @staticmethod
    def load_train_data_from_file(patient_name, filename,params=None):
        """
        Loading single file training data
        filename: full path to .mat file 
        return: (EEG data Instance, y_seizure, y_early)
        """
        #print "\nLoading train data for " + patient_name + filename
        eeg_data_tmp = EEGData(filename)
        # a list of Instance's
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1

        eeg_data = eeg_data[0]
        # eeg_data is now an Instance
        # determine labels based on filename
        if filename.find('interictal') > -1:
            y_seizure=0
            y_early=0
        elif eeg_data.latency < 15:
            y_seizure=1
            y_early=1
        else:
            y_seizure=1
            y_early=0

        fs = eeg_data.sample_rate

        # preprocessing
        data = eeg_data.eeg_data
        params['fs']=fs


        eeg_data.eeg_data = preprocessing.preprocess_multichannel_data(data, params)
        return (eeg_data, y_seizure, y_early)

    @staticmethod
    def load_test_data_from_file(patient_name, filename,params=None):
        """
        Loading single file test data
        :return: EEG data Instance (no labels returned)
        """
        assert ( filename.find('test'))
        #print "\nLoading test data for " + patient_name + filename
        eeg_data_tmp = EEGData(filename)
        eeg_data = eeg_data_tmp.get_instances()
        assert len(eeg_data) == 1
        # an Instance
        eeg_data = eeg_data[0]
        fs = eeg_data.sample_rate
        data = eeg_data.eeg_data
        params['fs']=fs

        eeg_data.eeg_data = preprocessing.preprocess_multichannel_data(data,params)
        return eeg_data


__author__ = 'Matthieu'

from Instance_pred import Instance
import scipy.io
import numpy as np

class EEGData(object):
    """
    Simple object that reads in the concatenated data, outputs slices and instances.

    path : path to the stitched (concatenated) data.


    eeg_data : the actual eeg_data. 2d array of channels x time
    sampling_rate : sampling rate of the data
    sequence : latency in tens of seconds in seconds [0 - 5]
    number_of_channels : number of channels in the data

    """

    def __init__(self, path):
        full_data = scipy.io.loadmat(path)


        samples = full_data.keys()
        for s in samples:
            if 'segment' in s:
                sample = s

        # base structure
        base_data = full_data[sample][0][0]
        self.eeg_data = base_data[0]

        var_row = np.std(self.eeg_data, axis=1)
        idx_zero_variance = var_row == 0
        n_rows_zero_variance = np.sum(idx_zero_variance)
        # NOTE: adding low variance noise to zero variance rows
        #           to prevent divide by zero errors elsewhere
        self.eeg_data[idx_zero_variance, :] = self.eeg_data[idx_zero_variance, :] \
            + 1e-6 * np.random.randn(n_rows_zero_variance, self.eeg_data.shape[1])

        # sequence

        if len(base_data) > 4: # sequence information only in test data
            self.sequence = base_data[4]
        else:
            self.sequence = 0
        # data_length_sec
        self.data_length_sec = base_data[1]
        # sampling_frequency
        self.sampling_frequency = base_data[2]
        self.sampling_rate = self.sampling_frequency

        # channels
        self.channels = base_data[3]

        self.number_of_channels = self.channels.shape[0]

        if 'interictal' in path:
            self.label = 0
        else:
            self.label = 1

        if "Dog_" in path:
            self.patient_type = "Dog"
            startIndex = path.find("Dog_")+4
            endIndex = path[startIndex:].find("_")
            self.patient_id = path[startIndex:startIndex+endIndex]
        else:
            self.patient_type = "Patient"
            startIndex = path.find("Patient_")+8  # 2 patients
            endIndex = path[startIndex:].find("_")
            self.patient_id = path[startIndex:startIndex+endIndex]


    def get_instances(self):
        instancesList = list()
        instance = Instance(self.patient_id, self.sequence, self.eeg_data, self.sampling_rate, self.number_of_channels)
        instancesList.append(instance)
        return instancesList

if __name__ == "__main__":
    path = '/Users/Matthieu/Dev/seizureDectectionKaggle/Dog_1_test_segment_1.mat'
    #path = '/home/nuke/git/gatsby-hackathon-seizure/wj_data/Dog_1/Dog_1_ictal_segment_1.mat'
    res = EEGData(path)
    test = res.get_instances()
    print test[0].eeg_data
    print "EEG Data length: %d" % (len(test))
    print type(test[0])

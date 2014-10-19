__author__ = 'Matthieu'

from Instance import Instance
import scipy.io
import h5py
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


        sample = full_data.keys()[0]
        base_data = full_data[sample][0][0]  # base structure

        self.eeg_data = base_data[0]
        var_row = np.std(self.eeg_data, axis=1)
        idx_zero_variance = var_row == 0
        n_rows_zero_variance = np.sum(idx_zero_variance)
        # NOTE: adding low variance noise to zero variance rows 
        #           to prevent divide by zero errors elsewhere
        self.eeg_data[idx_zero_variance, :] = self.eeg_data[idx_zero_variance, :] \
            + 1e-6 * np.random.randn(n_rows_zero_variance, self.eeg_data.shape[1])

        # sequence
        self.sequence = base_data[4]
        # data_length_sec
        self.data_length_sec = base_data[1]
        # sampling_frequency
        self.sampling_frequency = base_data[2]
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

    def get_time_channel_slice(self, channels=None, low_minute=None, high_minute=None):
        if not channels:
            channels = np.arange(self.number_of_channels)
        else:
            channels = np.array(channels)
        low_minute = low_minute or self.sequence[0]
        high_minute = high_minute or self.sequence[-1]
        if self.sampling_rate == self.eeg_data.shape[1]:
            sliced_data = self.eeg_data
        else:
            sliced_data = self.eeg_data[channels, (low_minute*10*self.sampling_rate*60):(high_minute*10*self.sampling_rate*60-1)]
        return sliced_data

    def get_instances(self):
        instancesList = list()
        if len(self.sequence) == 1:
            sliced_data = self.get_time_channel_slice(None, self.sequence, self.sequence+1)
            instance = Instance(self.patient_id, self.sequence, sliced_data, self.sampling_rate, self.number_of_channels)
            instancesList.append(instance)
        else:
            for tensOfMinutes in self.sequence[0:-1]:
                sliced_data = self.get_time_channel_slice(None, tensOfMinutes, tensOfMinutes+1)
                instance = Instance(self.patient_id, tensOfMinutes, sliced_data, self.sampling_rate, self.number_of_channels)
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

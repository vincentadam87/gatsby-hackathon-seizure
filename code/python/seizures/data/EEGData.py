__author__ = 'Matthieu'

from Instance import Instance
import scipy.io
import numpy as np
import copy

class EEGData(object):
    """
    Simple object that reads in the concatenated data, outputs slices and instances.

    path : path to the stitched (concatenated) data.


    eeg_data : the actual eeg_data
    sampling_rate : sampling rate of the data
    latency : latency in seconds [0 - 31]
    number_of_channels : number of channels in the data

    """

    def __init__(self, path):
        full_data = scipy.io.loadmat(path)
        self.eeg_data = full_data['data']

        if 'latency' in full_data.keys():
            if len(full_data['latency'].shape) == 1:
                self.latency = np.array(full_data['latency'])
            else:
                self.latency = np.array(full_data['latency'])[0, :]

        if 'freq' in full_data.keys():
            if len(full_data['freq'].shape) == 1:
                self.sampling_rate = round(full_data['freq'])
            else:
                self.sampling_rate = round(full_data['freq'][0, 0])

        if (not 'freq' in full_data.keys()) & ('latency' in full_data.keys()):
            self.sampling_rate = self.eeg_data.shape[1]/self.latency[-1]

        if (not 'latency' in full_data.keys()) & ('freq' in full_data.keys()):
            self.latency = np.array(range(int(round(self.eeg_data.shape[1]/self.sampling_rate))))

        self.number_of_channels = self.eeg_data.shape[0]

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
            startIndex = path.find("Patient_")+8
            endIndex = path[startIndex:].find("_")
            self.patient_id = path[startIndex:startIndex+endIndex]

    def get_time_channel_slice(self, channels=None, low_second=None, high_second=None):
        if not channels:
            channels = np.array(range(self.number_of_channels))
        else:
            channels = np.array(channels)

        low_second = low_second or self.latency[0]
        high_second = high_second or self.latency[-1]

        sliced_data = self.eeg_data[channels-1, (low_second*self.sampling_rate):(high_second*self.sampling_rate-1)]
        return sliced_data

    def get_instances(self):
        instancesList = list()

        if len(self.latency) == 1:
            sliced_data = self.get_time_channel_slice(None, self.latency, self.latency+1)
            instance = Instance(self.patient_id, self.latency, sliced_data, self.sampling_rate, self.number_of_channels)
            instancesList.append(instance)
        else:
            for second in self.latency[0:-1]:
                sliced_data = self.get_time_channel_slice(None, second, second+1)
                instance = Instance(self.patient_id, second, sliced_data, self.sampling_rate, self.number_of_channels)
                instancesList.append(instance)

        return instancesList


if __name__ == "__main__":
    path = '/Users/Matthieu/Dev/seizureDectectionKaggle/Dog_1_test_segment_1.mat'
    res = EEGData(path)
    test = res.get_instances()
    print test[0].eeg_data

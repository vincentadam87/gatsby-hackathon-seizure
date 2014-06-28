__author__ = 'Matthieu'


import scipy.io
import numpy as np

class EEGData:
    """
    Simple object that contains the EEG_data

    eeg_data : the actual eeg_data
    data_per_second : sampling rate of the data
    latency : latency in seconds [0 - 31]
    number_of_channels : number of channels in the data

    """
    def __init__(self, path):
        full_data = scipy.io.loadmat(path)
        self.eeg_data = full_data['data']
        self.latency = np.array(full_data['latency'])[0, :]
        self.data_per_second = self.eeg_data.shape[1]/len(self.latency)
        self.number_of_channels = self.eeg_data.shape[0]

    def get_time_channel_slice(self, channels=None, low_second=None, high_second=None):
        if not channels:
            channels = range(self.number_of_channels)
        else:
            channels = np.array(channels)-1

        low_second = low_second or self.latency[0]
        high_second = high_second or self.latency[-1]

        sliced_data = self.eeg_data[channels-1, (low_second*self.data_per_second):(high_second*self.data_per_second-1)]
        return sliced_data

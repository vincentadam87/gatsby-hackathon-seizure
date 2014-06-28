__author__ = 'Matthieu'


import scipy.io
import numpy as np
import matplotlib.pyplot as plt

class EEGData:
    """
    Simple object that contains the EEG_data

    path : path to the stitched (concatenated) data.


    eeg_data : the actual eeg_data
    sampling_rate : sampling rate of the data
    latency : latency in seconds [0 - 31]
    number_of_channels : number of channels in the data

    """

    def __init__(self, path):
        full_data = scipy.io.loadmat(path)
        self.eeg_data = full_data['data']
        if 'latency' in full_data:
            self.latency = np.array(full_data['latency'])[0, :]
        if 'freq' in full_data:
            self.sampling_rate = round(full_data['freq'][0, 0])
        self.number_of_channels = self.eeg_data.shape[0]

    def get_time_channel_slice(self, channels=None, low_second=None, high_second=None):
        if not channels:
            channels = np.array(range(self.number_of_channels))
        else:
            channels = np.array(channels)

        low_second = low_second or self.latency[0]
        high_second = high_second or self.latency[-1]

        sliced_data = self.eeg_data[channels-1, (low_second*self.sampling_rate):(high_second*self.sampling_rate-1)]
        return sliced_data
    
    def subsample_data(self, new_sampling_rate):
        """
        Subsamples the data to a new (lower) sampling rate. 
        Make sure the current sampling rate is a multiple of the new sampling rate!
    
        """
        
        subsampling_intervals = self.sampling_rate / new_sampling_rate
        new_length = self.eeg_data.shape[1] / subsampling_intervals
        new_eeg_data = np.empty((self.number_of_channels, new_length))
        
        for channel_index in range(0, self.number_of_channels):
            new_eeg_data[channel_index, :] = np.mean(self.eeg_data[channel_index, :].reshape(-1, subsampling_intervals), 1)
            
        self.eeg_data = new_eeg_data
        self.sampling_rate = new_sampling_rate




    def get_instances(self):
        instancesList = list()
        for second in self.latency[0:-1]:
            instancesList.append(self.get_time_channel_slice(None, second, second+1))

        return instancesList
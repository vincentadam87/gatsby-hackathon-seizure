import copy
import numpy as np

class Instance(object):
    """
    A simple data structure for one training instance.
    """
    def __init__(self, patient_id, latency, eeg_data, sample_rate, number_of_channels):
        self.patient_id = patient_id
        self.latency = latency
        #2d array of #channels x time 
        self.eeg_data = eeg_data
        # Two attribute names were inconsistently used. For backward
        # compatibility,  use both for now.
        self.sample_rate = sample_rate
        self.sampling_rate = sample_rate
        self.number_of_channels = number_of_channels


    def subsample_data(self, new_sampling_rate):
        """
        Subsamples the data to a new (lower) sampling rate and returns a new
        EEGData class instance with subsampled data. Make sure the current
        sampling rate is a multiple of the new sampling rate!
        """

        assert(new_sampling_rate <= self.sample_rate)

        if new_sampling_rate == self.sample_rate:
            return self

        new_eeg_data = copy.deepcopy(self)
        subsampling_intervals = new_eeg_data.sample_rate / new_sampling_rate
        new_length = new_eeg_data.eeg_data.shape[1] / subsampling_intervals
        new_eeg_data.eeg_data = np.empty((new_eeg_data.number_of_channels, new_length))

        for channel_index in range(0, new_eeg_data.number_of_channels):
            new_eeg_data.eeg_data[channel_index, :] = np.mean(self.eeg_data[channel_index, :].reshape(-1, subsampling_intervals), 1)

        new_eeg_data.sampling_rate = new_sampling_rate
        new_eeg_data.sample_rate = new_sampling_rate

        return new_eeg_data

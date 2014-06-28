from scipy.fftpack import fft, fftshift

import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase


class FFTFeatures(FeatureExtractBase):
    """
    Class to extracts spectral powers based on FFT features.
    @author Julian
    """

    def __init__(self, sampling_rate=400, bins=10):
        self.sampling_rate = sampling_rate
        self.bins = bins

    def extract(self, eeg_data_instance):
        subsampled_eeg_data = eeg_data_instance.subsample_data(self.sampling_rate)
        features = np.empty((subsampled_eeg_data.number_of_channels, self.bins))

        for channel_index in range(0, subsampled_eeg_data.number_of_channels):
            frequencies = abs(fftshift(fft(subsampled_eeg_data.eeg_data[channel_index, :])))
            frequencies_to_sum = len(frequencies) / (self.bins * 2)

            for i in range(1, self.bins):
                features[channel_index, i] = np.mean(np.square(frequencies[(i - 1) * frequencies_to_sum:i * frequencies_to_sum]))

        return features

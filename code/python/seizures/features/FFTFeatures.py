from scipy.fftpack import fft, fftshift

import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase


class FFTFeatures(FeatureExtractBase):
    """
    Extracts spectral powers based on FFT features
    
    @author Julian
    """

    def __init__(self, bins):
        self.bins = bins

    def extract(self, eeg_data_instance):
        features = np.empty((eeg_data_instance.number_of_channels, self.bins))

        for channel_index in range(0, eeg_data_instance.number_of_channels):
            frequencies = abs(fftshift(fft(eeg_data_instance.eeg_data[channel_index, :])))
            frequencies_to_sum = len(frequencies) / (self.bins * 2)

            for i in range(1, self.bins):
                features[channel_index, i] = np.mean(np.square(frequencies[(i - 1) * frequencies_to_sum:i * frequencies_to_sum]))

        return features

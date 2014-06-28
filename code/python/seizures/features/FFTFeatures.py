__author__ = 'Julian'

import scipy.io
import numpy as np
from scipy.fftpack import fft, ifft, fftshift

from seizures.features.FeatureExtractBase import FeatureExtractBase


class FFTFeatures(FeatureExtractBase):
    """
    Extracts spectral powers based on FFT features
    """

    def __init__(self, data, bins):
        self.data = data
        self.bins = bins

    def extract(self):
        features = np.empty((self.data.number_of_channels, self.bins))

        for channel_index in range(0, self.data.number_of_channels):
            frequencies = abs(fftshift(fft(self.data.eeg_data[channel_index,:])))
            frequencies_to_sum = len(frequencies) / (self.bins * 2)

            for i in range(1, self.bins):
                features[channel_index, i] = np.mean(np.square(frequencies[(i-1)*frequencies_to_sum:i*frequencies_to_sum]))

        return features

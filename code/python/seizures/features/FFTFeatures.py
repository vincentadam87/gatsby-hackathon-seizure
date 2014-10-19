from scipy.fftpack import fft, fftshift

import numpy as np
import math
from seizures.features.FeatureExtractBase import FeatureExtractBase


def nextpow2(i):
    #n = 1
    #while n < i: n *= 2
    #return n
    return int(2**math.ceil(math.log(i)/math.log(2)))

class FFTFeatures(FeatureExtractBase):
    """
    Class to extracts spectral powers based on FFT features.
    @author Julian
    """

    def __init__(self, band_means, band_width):
        self.band_means = band_means
        self.band_width = band_width

    def extract(self, instance):
        data = instance.eeg_data
        L = data.shape[1]

        nfft = nextpow2(L)
        Y = fft(data, nfft)
        psd = 2 * abs(Y[:,:(Y.shape[1] / 2)]) ** 2
        f = instance.sample_rate / 2 * np.linspace(0, 1, nfft / 2)

        feats = np.zeros((instance.number_of_channels, len(self.band_means)))
        for i in range(len(self.band_means)):
            inds = abs(self.band_means[i] - f) < self.band_width
            feats[:, i] += np.sum(psd[:,inds], 1)

        return feats.reshape((feats.shape[0]*feats.shape[1],))


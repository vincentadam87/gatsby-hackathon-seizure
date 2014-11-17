
from scipy.fftpack import fft
import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase


class SpectralBandsFeatures(FeatureExtractBase):
    """
    Class to extracts spectral Energy features.
    In this case we focus on the power spectra from EEG-relevant frequency bands
    bands: (rho,theta,alpha,beta,gamma)
    rho:    0.5-4 Hz
    theta:    4-8 Hz
    alpha:    8-13 Hz
    beta:    13-30 Hz
    gamma:   30-48 Hz
    @author Joana and Vincent
    """

    def __init__(self, bands_edges=[[0.5,4],[4,8],[8,13],[13,30],[30,48]]):
        self.bands_edges = bands_edges

    def extract(self, instance):
        # -----------------
        data = instance.eeg_data
        n_ch,time = data.shape
        fs = instance.sample_rate

        # shared frequency axis
        freqs = np.fft.fftfreq(time)
        # spectral density per band
        SEdata = np.abs(np.fft.fft(data,axis=1))**2

        If = range(len(freqs))
        n_band = len(self.bands_edges)

        bands = []
        for band_edge in self.bands_edges:
            bands.append( [i for i in I if (freqs[i]*fs >band_edge[0])&(freqs[i]*fs <band_edge[1])] )

        # creating features
        features = np.zeros((n_ch,n_band))
        i_band = 0
        for band in bands:
            features[:,i_band] = np.mean(SEdata[:,band],axis=1)
            i_band+=1
        return np.hstack(features)

    def __str__(self):
        return "SpectralBandsFeatures" + '(%sHz)'% (', '.join(str(b) for b in self.bands_edges))


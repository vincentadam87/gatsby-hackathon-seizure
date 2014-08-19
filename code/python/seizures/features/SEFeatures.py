from scipy.fftpack import fft, fftshift

import numpy as np
import math
from seizures.features.FeatureExtractBase import FeatureExtractBase
from scipy.signal import kaiserord, firwin



class SEFeatures(FeatureExtractBase):
    """
    Class to extracts spectral Energy  features.
    @author Vincent
    """

    def __init__(self):
        pass

    def extract(self, instance):
        # -----------------
        data = instance.eeg_data
        n_ch,time = data.shape
        fs = instance.sample_rate

        # shared frequency axis
        freqs = np.fft.fftfreq(time)
        # spectral density per band
        SEdata = np.real(np.fft.fft(data,axis=1))**2

        I = range(time)
        fmax = 200.
        If = [i for i in I if freqs[i]*fs <fmax] # indices of freq below X Hz

        NBAND = 60
        L = len(If)/min(NBAND,len(If)) # cutting spectrum in homogenous bands
        edges = If[0::L] # bands of length L
        bands = [range(edges[i],edges[i+1]) for i in range(len(edges)-1) ]
        n_band = len(bands)

        # creating features
        features = np.zeros((n_ch,n_band))
        i_band = 0
        for band in bands:
            features[:,i_band] = np.mean(SEdata[:,band],axis=1)
            i_band+=1
        return np.hstack(features)

    def __str__(self):
        return "SE"



import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.Instance import Instance
import numpy as np
from numpy import unwrap, angle
from scipy.signal import hilbert
from matplotlib import pyplot as plt

class PLVFeatures(FeatureExtractBase):
    """
    Class to extracts Phase Locking Value (PLV) between pairs of channels.
    @author V&J
    """

    def __init__(self):
        pass

    def extract(self, instance):
        data = instance.eeg_data
        n_ch, time = data.shape
        n_pairs = n_ch*(n_ch-1)/2
        # initiate matrices
        phases = np.zeros((n_ch,time))
        delta_phase_pairwise = np.zeros((n_pairs,time))
        plv = np.zeros((n_pairs,))

        # extract phases for each channel
        for c in range(n_ch):
            phases[c,:] = unwrap(angle(hilbert(data[c,:])))

        # compute phase differences
        k = 0
        for i in range(n_ch):
            for j in range(i+1,n_ch):
                delta_phase_pairwise[k,:] = phases[i,:]-phases[j,:]
                k+=1

        # compute PLV
        for k in range(n_pairs):
            plv[k] = np.abs(np.sum(np.exp(1j*delta_phase_pairwise[k,:]))/time)

        self.assert_features(plv)
        # features = a 1d ndarray
        return plv

    def __str__(self):
        return "PLV"





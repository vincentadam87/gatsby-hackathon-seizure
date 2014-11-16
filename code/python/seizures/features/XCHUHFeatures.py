import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
import scipy.signal

class XCHUHFeatures(FeatureExtractBase):
    """
    Class to build features from channel pairwise cross correlation
    @author Ben
    """

    def __init__(self, N_singular_vec=10,N_fft=30):
        self.N_singular_vec = N_singular_vec
        self.N_fft = N_fft

    def extract(self, instance ):
        data = instance.eeg_data
        sample_rate =instance.sample_rate
        q = int(sample_rate/400) # ds facto
        data_ds	= scipy.signal.decimate(data, q)
        #building pairwise cross correlations
        N_time = data_ds.shape[1]
        N_channel=data_ds.shape[0]
        M = np.zeros( (N_channel*(N_channel-1)/2, 2*N_time-1) )
        i = 0
        for v in range(N_channel):
            for w in range(0,v):
                M[i,:] = np.correlate(data_ds[v],data_ds[w],mode="full")
                i+=1
        #compution SVD
        U, s, V = np.linalg.svd(M, full_matrices=True)
        sgn=np.sign(V[:,(V.shape[1]-1)/2])
        V_sgn=np.dot(np.diag(sgn),V)
        V_fft = np.real(np.fft.fft(V_sgn[0:self.N_singular_vec,:]))
        return np.hstack(V_fft[:,0:self.N_fft])

    def __str__(self):
        return "XCHUH"+'(%d,%d)'% (self.N_singular_vec, self.N_fft)



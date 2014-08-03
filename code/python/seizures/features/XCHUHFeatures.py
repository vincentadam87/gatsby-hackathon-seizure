import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
import scipy.signal

class XCHUHFeatures(FeatureExtractBase):
    """
    Class to build features from channel pairwise cross correlation
    @author Ben
    """

    def __init__(self):
        pass

    def extract(self, instance,N_singular_vec=10,N_fft=30 ):
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
	V_fft = np.real(np.fft.fft(V_sgn[0:N_singular_vec,:]))
	return np.hstack(V_fft[:,0:N_fft])


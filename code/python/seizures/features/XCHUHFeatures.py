import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from statsmodels.tsa.vector_ar.var_model import VAR
import scipy.signal
from itertools import tee, izip
class XCHUHFeatures(FeatureExtractBase):
    """
    Class to extracts AR(2) features.
    @author V&J
    """

    def __init__(self):
        pass

    def extract(self, instance):
        data = instance.eeg_data
        q = 2 # ds facto		
  		data_ds	= scipylsignal.decimate(data, q)

		def pairwise(iterable):
	    	"s -> (s0,s1), (s1,s2), (s2, s3), ..."
	    	a, b = tee(iterable)
	    	next(b, None)
	    	return izip(a, b)

	    #building pairwise cross correlations
	    N_time = data_ds.shape[1]
	    N_channel=data_ds.shape[0]
	    M = np.zeros( (N_channel*(N_channel-1)/2, 2*N_time-1) )
		i = 0
		for v, w in pairwise(range(N_channel)):
			M[i,:] = np.correlate(data_ds[v],data_ds[w],mode="full")
			i+=1

		#compution SVD
		U, s, V = np.linalg.svd(M, full_matrices=True)

		N_singular_vec = 10
		N_fft = 30
		V_fft = np.abs(np.fft.fft(V[:,0:N_singular_vec-1]))[0:N_fft-1,:]

		#final extraction
		return np.hstack(V_fft)
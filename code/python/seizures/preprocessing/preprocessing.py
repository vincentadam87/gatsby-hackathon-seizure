import numpy as np
from scipy.signal import firwin, kaiserord, convolve2d, decimate
#from matplotlib import pyplot as plt

# DEFINE FILTERS FOR PREPROCESSING:
def preprocess_multichannel_data(matrix,params):

    """
    :param matrix: multichannel EEG data
    :param fs: sampling frequency
    :return: data without mains, electrical artefacts, aliasing

    authors: Lea and Vincent
    """
    assert(type(matrix)==np.ndarray)
    #print 'downsample...'
    matrix,fs = downsample(matrix,params)
    params['fs']=fs # update sampling rate
    #print 'initial ', matrix.shape
    matrix = remove_elec_noise(matrix,params)
    #print 'elec noise ', matrix.shape
    matrix = anti_alias_filter(matrix,params)
    #print 'anti alias ', matrix.shape
    matrix = remove_dc(matrix)
    #print 'dc ', matrix.shape
    return matrix

def downsample(matrix,params):

    """
    :param matrix: multichannel EEG data
    :param params: takes in sampling frequency fs from params dict
    :return: downsampled data; downsample to dogfrequency but maximum downsampling rate set to 8  
    """
    fs = int(params['fs'])
    targetrate = int(params['targetrate'])
    dsfactor = max(1,int(fs/targetrate)) #should come out to 1, i.e. no downsampling for dogs
    #print 'dsfactor calculated =', dsfactor
    #maxdsfactor = int(8)

    #if dsfactor > maxdsfactor:
    #   dsfactor = maxdsfactor
    #print 'dsfactor used =', dsfactor
    if dsfactor>1:
        ds_matrix = decimate(matrix,dsfactor,axis=1)
        return ds_matrix, float(fs/dsfactor)
    else:
        return matrix, float(fs)

    #ds_list=[]
    #for i in range(matrix.shape[0]):
    #    x = matrix[i,:]
    #    ds_x = decimate(x, dsfactor)
    #    ds_list.append(ds_x)
    #ds_matrix = np.asarray(ds_list)



def remove_dc(x):
    #print x.shape
    assert(type(x)==np.ndarray)
    """
    Remove mean of signal
    :return: x - mean(x)
    """
    x_dc = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_dc[i,:] = x[i,:] - np.mean(x[i,:])
    return x_dc

# build custom filter; inspiration : http://wiki.scipy.org/Cookbook/FIRFilter
def build_filter(fs,cutoff,width,attenuation):
    """
    Building low pass filter impulse response0
    :param fs: sampling rate
    :return: 1D array impulse response of lp filter
    """
    # define params:
    nyq_rate = fs / 2.0 # Nyquist frequency
    if type(cutoff)==list:
        cutoff = [c/nyq_rate for c in cutoff]
    else:
        cutoff=cutoff/nyq_rate

    # Compute the order and Kaiser parameter for the FIR filter:
    N, beta = kaiserord(attenuation, width/nyq_rate)

    # Use firwin with a Kaiser window to create a lowpass FIR filter:

    if N%2==0:
        N+=1# odd trick

    taps = firwin(N, cutoff, window=('kaiser', beta),pass_zero=True)

    return taps

def remove_elec_noise(x,params):
    fs = params['fs']

    # import the relevant filters from signal processing library
    assert(type(x)==np.ndarray)

    cutoff = params['elec_noise_cutoff']
    width = params['elec_noise_width']
    attenuation = params['elec_noise_attenuation']
    f = build_filter(fs,cutoff,width,attenuation)
    f = np.expand_dims(f,axis=0)
    #print x.shape, f.shape
    filtered_x = convolve2d(x,f,mode='same') # NB: mode='same' cuts beginning & end

    return filtered_x

def anti_alias_filter(x,params):

    """
    Anti_aliasing: use Nyquist frequ cutoff low-pass filter
    :return: anti-aliased signal
    """
    fs = params['fs']

    assert(type(x)==np.ndarray)

    # def build_aa_filter(fs):
    #     """
    #     :param fs: sampling rate
    #     :return: 1D array impulse response using nyq frequ filter
    #     """
    #     # define params:
    #     nyq_rate = fs / 2.0 # Nyquist frequency
    #     width = 1000.0/nyq_rate # width of the transition from pass to stop relative to the Nyquist rate; here: 5 Hz
    #     ripple_db = 120.0 # attenuation in the stop band, in dB.
    #     cutoff_hz = nyq_rate - 1 # cutoff freq
    #
    #     # Compute the order and Kaiser parameter for the FIR filter:
    #     N, beta = kaiserord(ripple_db, width)
    #
    #     # Use firwin with a Kaiser window to create a lowpass FIR filter:
    #     taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    #
    #     return taps
    cutoff = params['anti_alias_cutoff']
    width = params['anti_alias_width']
    attenuation = params['anti_alias_attenuation']

    if (cutoff==None) or (cutoff>=fs/2.):
        return x
    else:
        f = build_filter(fs,cutoff,width,attenuation)
        f = np.expand_dims(f,axis=0)
        filtered_x = convolve2d(x,f,mode='same') # NB: mode='same' cuts beginning & end
        return filtered_x


import numpy as np
from scipy.signal import firwin, kaiserord, convolve2d
from scipy.signal import convolve as conv
#from matplotlib import pyplot as plt

# DEFINE FILTERS FOR PREPROCESSING:
def preprocess_multichannel_data(matrix,fs):

    """
    :param matrix: multichannel EEG data
    :param fs: sampling frequency
    :return: data without mains, electrical artefacts, aliasing

    authors: Lea and Vincent
    """
    assert(type(matrix)==np.ndarray)

    print 'initial ', matrix.shape

    matrix = remove_elec_noise(matrix,fs)
    print 'elec noise ', matrix.shape
    matrix = anti_alias_filter(matrix,fs)
    print 'anti alias ', matrix.shape
    matrix = remove_dc(matrix)
    print 'dc ', matrix.shape
    return matrix

def remove_dc(x):
    print x.shape
    assert(type(x)==np.ndarray)
    """
    Remove mean of signal
    :return: x - mean(x)
    """
    x_dc = np.zeros(x.shape)
    for i in range(x.shape[0]):
        x_dc[i,:] = x[i,:] - np.mean(x[i,:])
    return x_dc

def build_bp_filter(x,band,fs,attdb):
    """
    Building low pass filter impulse response0
    :param fs: sampling rate
    :return: 1D array impulse response of lp filter
    """
    # define params:
    nyq_rate = fs / 2.0 # Nyquist frequency
    width = 10. # width of the transition from pass to stop relative to the Nyquist rate; here: 5 Hz
    attdb = 40.0 # attenuation in the stop band, in dB.
    band = [49.,51.]

    nyq_rate = fs / 2.0 # Nyquist frequency
    cutoff_hz = [fr/nyq_rate for fr in band]

    # Compute the order and Kaiser parameter for the FIR filter:
    N, beta = kaiserord(attdb, width/nyq_rate)
    # Use firwin with a Kaiser window to create a lowpass FIR filter:
    taps = firwin(N, cutoff_hz, window=('kaiser', beta))

    return taps


def remove_elec_noise(x,fs):
    # import the relevant filters from signal processing library
    assert(type(x)==np.ndarray)

    # define params:
    nyq_rate = fs / 2.0 # Nyquist frequency
    width = 10. # width of the transition from pass to stop relative to the Nyquist rate; here: 5 Hz
    attdb = 40.0 # attenuation in the stop band, in dB.
    band = [49.,51.]

    f = build_bp_filter(x,band,fs,attdb)
    f = np.expand_dims(f,axis=0)
    #print 'matrix.shape ',matrix.shape, 'f.shape ', f.shape # check matrix and filter have same shape -> expand if necessary
    #print 'after expansion: ','matrix.shape ',matrix.shape, 'f.shape ', f.shape # check matrix and filter have same shape
    print x.shape, f.shape
    filtered_x = conv(x,f,mode='same') # NB: mode='same' cuts beginning & end

    return filtered_x



def build_aa_filter(fs):
        """
        :param fs: sampling rate
        :return: 1D array impulse response using nyq frequ filter
        """
        # define params:
        nyq_rate = fs / 2.0 # Nyquist frequency
        width = 1000.0/nyq_rate # width of the transition from pass to stop relative to the Nyquist rate; here: 5 Hz
        ripple_db = 120.0 # attenuation in the stop band, in dB.
        cutoff_hz = nyq_rate - 1 # cutoff freq

        # Compute the order and Kaiser parameter for the FIR filter:
        N, beta = kaiserord(ripple_db, width)

        # Use firwin with a Kaiser window to create a lowpass FIR filter:
        taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

        return taps


def anti_alias_filter(x,fs):
    """
    Anti_aliasing: use Nyquist frequ cutoff low-pass filter
    :return: anti-aliased signal
    """
    assert(type(x)==np.ndarray)

    f = build_aa_filter(fs)
    f = np.expand_dims(f,axis=0)

    filtered_x = convolve2d(x,f,mode='same') # NB: mode='same' cuts beginning & end

    return filtered_x


# DO PREPROCESSING:
# fs = 4000
# processed = preprocess_multichannel_data(matrix,fs)

# # # PLOT TO CHECK:
# # generate test matrix; check it looks remotely like the data
# matrix = np.random.random((4,1000))
# for i in range(matrix.shape[0]):
#     plt.plot( matrix[i,:]+i)
# plt.show()
#
# fs = 400
#
# # apply filters:
# matrix = remove_dc(matrix)
#
# for i in range(matrix.shape[0]):
#     plt.plot(matrix[i]+2*i)
# plt.show()
#
# for i in range(matrix.shape[0]):
#     plt.plot(np.abs(np.fft.fft(matrix[i]))[0:200]+2*i)
# plt.show()
#
# matrix = anti_alias_filter(matrix,fs)
#
# for i in range(matrix.shape[0]):
#     plt.plot(matrix[i]+2*i)
# plt.show()
#
# matrix = remove_elec_noise(matrix,fs)
#
# for i in range(matrix.shape[0]):
#     plt.plot(np.abs(np.fft.fft(matrix[i]))[0:200]+2*i)
# plt.show()
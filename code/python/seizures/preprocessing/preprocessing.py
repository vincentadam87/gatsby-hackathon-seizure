import scipy.signal as signal


def preprocess_multichannel_data(matrix,fs):


    """
    :param matrix: multichannel EEG data
    :param fs: sampling frequency
    :return: data without mains, electrical artefacts etc

    authors: Lea and Vincent
    """
    n_channel,m= matrix.shape
    for i in range(n_channel):
        preprocess_single_channel(matrix[i,:],fs)

def preprocess_single_channel(x,fs):
    x = remove_elec_noise(x,fs)
    x = anti_alias_filter(x)
    x = remove_dc(x)
    return x

def remove_dc(x):
    """
    Remove mean of signal: use 0.5Hz cut-off hp filter
    :return:
    """
    x = signal.medfilt(x)
    return x


def remove_elec_noise(x,fs):
    """
    Bandpass remove:59-61Hz (US); if data from EU/UK 49-51Hz
    :return:
    """
    bandstop = 60
    lowcut = bandstop-1
    highcut = bandstop+1


    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(x, lowcut, highcut, fs, order=5):
        b, a = signal.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    return butter_bandpass_filter(x,fs)



def anti_alias_filter(x,fs):
    """
    Anti_aliasing: use Nyquist frequ cutoff low-pass filter
    :return:
    """
    numtaps = 1
    cutoff = 0.5 * fs

    x = signal.firwin(numtaps, cutoff)

    return x

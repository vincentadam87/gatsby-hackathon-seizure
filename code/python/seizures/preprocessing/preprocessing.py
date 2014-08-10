import scipy.signal

def preprocess_multichannel_data(matrix):
    n_channel,m= matrix.shape
    for i in range(n_channel):
        preprocess_single_channel(matrix[i,:])

def preprocess_single_channel(x):
    x = remove_elec_noise(x)
    x = hp_filter(x)
    x = remove_dc(x)
    return x

def remove_dc():
    """
    Remove mean of signal
    :return:
    """
    pass

def remove_elec_noise():
    """
    Bandpass remove:49-51Hz
    :return:
    """
    pass

def hp_filter():
    """
    Anti_aliasing
    :return:
    """
    pass
import os,sys
import scipy.io, scipy.signal
import numpy as np
from matplotlib import pyplot as plt

path = '/home/vincent/Documents/Gatsby_hackathon/data/'


dir = os.path.realpath(path)
if dir not in sys.path:
    sys.path.insert(0, dir)
dir = os.path.realpath(path)
if dir not in sys.path:
    sys.path.insert(0, dir)

# sampling rates for human1, human2 and dog:
h1sr = 500
h2sr = 5000
dogsr = 400

def plot_data(filename,fs):
    datamat = scipy.io.loadmat(path+filename)
    data = datamat['data']
    Nchannel = data.shape[0]
    Nbin = data.shape[1]
    time = np.arange(Nbin)/fs

    fig = plt.figure(figsize=(20,3))
    X = np.arange(Nbin)
    for i_ch in range(Nchannel):
        s = np.std(data[i_ch,:])
        plt.plot(data[i_ch,:]/s+ 5*i_ch) #shift
    plt.show()

# filenames:

dog = 'Dog_1/Dog_1_ictal_segment_1.mat'

h1i1 = 'Patient_1_ictal_segment_1.mat'
h1i2 = 'Patient_1_ictal_segment_2.mat'

h1ii1 = 'Patient_1_interictal_segment_1.mat'

h2i1 = 'Patient_2_ictal_segment_1.mat'
h2i2 = 'Patient_2_ictal_segment_2.mat'
h2i3 = 'Patient_2_ictal_segment_3.mat'

h2ii1 = 'Patient_2_interictal_segment_1.mat'

fn = h2i3
datamat = scipy.io.loadmat(fn)
data = datamat['data']

h = scipy.signal.gaussian(50,10) # gaussian filter to get rid of electrical artefacts

N_channels = data.shape[0]
N_bins = data.shape[1]
q = int(h2sr/dogsr)
q=4
N_bins_ds = N_bins/q-1
print N_bins, N_bins_ds,q
data_ds = np.zeros((N_channels,N_bins_ds))
for i in np.arange(0,len(data)):
    x = data[i,:]
    x_d = scipy.signal.decimate(x, q) #downsample to 400 Hz. NB:scipy.signal.decimate(x,q) <- q is integer 
                                                  #i.e. human1 is not downsampled
    x_df = scipy.signal.convolve(x_d,h,mode='same') #filter
    data_ds[i,:] = x_df[0:N_bins_ds]

datamat_ds = datamat
datamat['data']=data_ds

import os,sys
import scipy.io, scipy.signal
import numpy as np
from matplotlib import pyplot as plt

dir = os.path.realpath('../data')
if dir not in sys.path:
    sys.path.insert(0, dir)
dir = os.path.realpath('../data/human_1/ictal/')
if dir not in sys.path:
    sys.path.insert(0, dir)

# sampling rates for human1, human2 and dog:
h1sr = 500
h2sr = 5000
dogsr = 400

def plot_data(filename):
    datamat = scipy.io.loadmat(filename)
    data = datamat['data']
    Nchannel = data.shape[0]
    Nbin = data.shape[1]
    fs = 5000. # for dog= 400; for human1=500; for human2=5000
    time = np.arange(Nbin)/fs

    fig = plt.figure(figsize=(20,3))
    X = np.arange(Nbin)
    for i_ch in range(Nchannel):
        s = np.std(data[i_ch,:])
        plt.plot(time, data[i_ch,:]/s) #+ 5*i_ch) #shift
    plt.show()

# filenames:

dog = '../data/Dog_1/Dog_1_ictal_segment_1.mat'

h1i1 = '../data/human_1/ictal/Patient_1_ictal_segment_1.mat'
h1i2 = '../data/human_1/ictal/Patient_1_ictal_segment_2.mat'

h1ii1 = '../data/human_1/interictal/Patient_1_interictal_segment_1.mat'

h2i1 = '../data/human_2/ictal/Patient_2_ictal_segment_1.mat'
h2i2 = '../data/human_2/ictal/Patient_2_ictal_segment_2.mat'
h2i3 = '../data/human_2/ictal/Patient_2_ictal_segment_3.mat'

h2ii1 = '../data/human_2/interictal/Patient_2_interictal_segment_1.mat'

# plot raw data:
plot_data(h2ii1)

# downsample, filter & plot data:
fn = h2ii1
datamat = scipy.io.loadmat(fn)
data = datamat['data']

h = scipy.signal.gaussian(50,10) # gaussian filter to get rid of electrical artefacts

for i in np.arange(0,len(data)):
    x = data[i,:]

    x_d = scipy.signal.decimate(x, h2sr/dogsr) #downsample to 400 Hz. NB:scipy.signal.decimate(x,q) <- q is integer 
                                               #i.e. human1 is not downsampled
    x_df = scipy.signal.convolve(x_d,h) #filter
    
    plt.plot(x_df)
    
plt.show()

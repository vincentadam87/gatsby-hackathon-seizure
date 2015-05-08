from seizures.Global import Global
from seizures.data.EEGData import EEGData
from seizures.preprocessing import preprocessing
import csv
import pickle
from statsmodels.tsa.vector_ar.var_model import VAR

patients = ['Dog_1','Dog_2','Dog_3','Dog_4',
            'Patient_1','Patient_2','Patient_3','Patient_4',
            'Patient_5','Patient_6','Patient_7','Patient_8']
# open csv of train
fname = '/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/data/train_filenames.txt'
with open(fname, 'rb') as csvfile:
     reader = csv.reader(csvfile, delimiter=',')
     flist = [row[0] for row in reader]

# iterate over segment file
root = Global.path_map('clips_folder_initial')
savpath = '/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/results/AR_fit/'

params = { 'anti_alias_cutoff': 500.,
              'anti_alias_width': 30.,
              'anti_alias_attenuation' : 40,
              'elec_noise_width' :3.,
              'elec_noise_attenuation' : 60.0,
              'elec_noise_cutoff' : [59.,61.],
              'targetrate':400}


# ----------------------


n = len(flist)
for i, f in enumerate(flist):

    patient = [p for p in patients if p in f][0]
    fpath = root + patient + '/' + f

    print fpath, float(i)/float(n)*100.


    # store fit with pickle
    eeg_data_tmp = EEGData(fpath)
    eeg_data = eeg_data_tmp.get_instances()
    assert len(eeg_data) == 1
    # eeg_data is now an Instance
    eeg_data = eeg_data[0]
    if f.find('interictal') > -1:
        latency=-1
    else:
        latency = eeg_data.latency

    fs = eeg_data.sample_rate
    # preprocessing
    data = eeg_data.eeg_data
    params['fs']=fs
    ### comment if no preprocessing
    eeg_data = preprocessing.preprocess_multichannel_data(data,params)
    for lag in [1,2]:
        prm = VAR(eeg_data.T)._estimate_var(lag).params
        savname = savpath + f.split('.')[0] + '_lag'+str(lag)+'.p'
        pickle.dump({'prm':prm,
                     'lag':lag,
                     'filename':f,
                     'latency':latency},open(savname, "wb" ))


# ----------------------------

# Now analysis:

# ICTAL(latency) vs INTERICTAL
# ======= (Per-patient)
# Unsupervised
# - do PCA on A matrices
# - do PCA on Q
# - do PCA on all
# Discriminative
# - Fisher discriminant analysis
# plot in low dim : color per class
# plot in low dim : color per latency
# ======= (Across-patient)
# Feature independent of number of channel (check variation with class and latency)
# - trace(Q)/nq (overall noise amplitude)
# - trace(A)/na (sum of eig)
# - eig(A)/sum(eig(A)): normalized eigenvalue spectrum
# - mean(|diag(A)|)/mean(|Offdiag(A)|)


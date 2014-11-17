from seizures.preprocessing.PreprocessingBase import PreprocessingBase
from seizures.preprocessing.preprocessing_utils import preprocess_multichannel_data

import numpy as np
from scipy.signal import firwin, kaiserord, convolve2d, decimate

class PreprocessingLea(PreprocessingBase):
    """
    The initial preprocessing method used for the earlier competition
    """

    params = None

    def __init__(self, **kwargs):
        if ('params' not in kwargs):
            self.params =  {'anti_alias_attenuation' : 40,
          'elec_noise_width' :3.,
          'elec_noise_attenuation' : 60.0,
          'elec_noise_cutoff' : [59.,61.],
          'targetrate':400}
          #'anti_alias_cutoff': 500.,
          #'anti_alias_width': 30.,

        else:
            self.params = kwargs['params']

        if 'targetrate' in kwargs:
            self.params['targetrate'] = kwargs['targetrate']
        if 'anti_alias_cutoff' in kwargs:
            self.params['anti_alias_cutoff'] = kwargs['anti_alias_cutoff']



    def apply(self, X, fs):
        tmp_params = self.params
        tmp_params['fs'] = fs
        return preprocess_multichannel_data(X, tmp_params)

    def __str__(self):
        return str(self.params)+""
import copy
import numpy as np

class Instance(object):
    """
    A simple data structure for one training instance.
    """
    def __init__(self, patient_id, sequence, eeg_data, sample_rate, number_of_channels):
        self.patient_id = patient_id
        self.sequence = sequence
        #2d array of #channels x time 
        self.eeg_data = eeg_data
        # Two attribute names were inconsistently used. For backward
        # compatibility,  use both for now.
        self.sample_rate = sample_rate
        self.sampling_rate = sample_rate
        self.number_of_channels = number_of_channels


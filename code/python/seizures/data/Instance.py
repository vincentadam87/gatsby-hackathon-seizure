class Instance(object):
    """
    A simple data structure for one training instance.

    eeg_data - 2d numpy array of channels x time

    @author: Shaun
    """
    
    def __init__(self, patient_id, latencies, eeg_data, sample_rate):
        self.patient_id = patient_id
        self.latencies = latencies
        self.eeg_data = eeg_data
        self.sample_rate = sample_rate
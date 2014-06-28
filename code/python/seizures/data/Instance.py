class Instance(object):
    """
    A simple data structure for one training instance.
    """
    def __init__(self, patient_id, latency, eeg_data, sample_rate):
        self.patient_id = patient_id
        self.latency = latency
        self.eeg_data = eeg_data
        self.sample_rate = sample_rate
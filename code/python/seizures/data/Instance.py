
class Instance(object):
    """
    A simple data structure for one training instance.
    """


    # def __init__(self, patient_id, sequence, eeg_data, sample_rate, number_of_channels):
    #     self.patient_id = patient_id
    #     self.sequence = sequence
    #     #2d array of #channels x time
    #     self.eeg_data = eeg_data
    #     # Two attribute names were inconsistently used. For backward
    #     # compatibility,  use both for now.
    #     self.sample_rate = sample_rate
    #     self.sampling_rate = sample_rate
    #     self.number_of_channels = number_of_channels

    def __init__(self, *args, **kwargs):

        if 'instance' in kwargs:

            self.patient_id = kwargs['instance'].patient_id
            self.sequence = kwargs['instance'].sequence
            # Two attribute names were inconsistently used. For backward
            # compatibility,  use both for now.
            self.sample_rate = kwargs['instance'].sample_rate
            self.sampling_rate = kwargs['instance'].sample_rate
            self.number_of_channels = kwargs['instance'].number_of_channels
            if 'eeg_data' in kwargs['instance']:
                #2d array of #channels x time
                self.eeg_data = kwargs['eeg_data']
            else:
                self.eeg_data = kwargs['instance'].eeg_data

        else:
            self.patient_id = args[0]
            self.sequence = args[1]
            #2d array of #channels x time
            self.eeg_data = args[2]
            # Two attribute names were inconsistently used. For backward
            # compatibility,  use both for now.
            self.sample_rate = args[3]
            self.sampling_rate = args[3]
            self.number_of_channels = args[4]



from seizures.features.FeatureExtractBase import FeatureExtractBase
import numpy as np
from seizures.data.Instance import Instance

class FeatureSplitAndAverage(FeatureExtractBase):
    """
    FeatureSplitAndAverage : split data in k , compute features and average
    @author V&J
    """

    def __init__(self, feature, k=10):
        assert(isinstance(feature, FeatureExtractBase))
        self.feature = feature
        self.k = k

    def extract(self, instance):

        data = instance.eeg_data
        n_ch,time = data.shape

        L = int(time/self.k)
        extracted_features_list = []

        # select sub data

        for i in range(self.k):
            if i == self.k-1: # if last
                sub_data = data[:,i*L:-1]
            else:
                sub_data = data[:,i*L:(i+1)*L]

            # apply feature on subdata
            sub_instance = Instance(instance=instance, eeg_data=sub_data)
            feature = self.feature.extract(sub_instance)
            extracted_features_list.append(np.hstack(feature))

        return sum(extracted_features_list)/self.k







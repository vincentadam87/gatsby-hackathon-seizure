import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.RandomFeatures import RandomFeatures
from seizures.features.XCHUHFeatures import XCHUHFeatures

class MixFeatures(FeatureExtractBase):
    """
    Class to concatenate output of individual feature classes.
    @author V&J
    """

    def __init__(self, features_list):
        """
        Wittawat: features_list is a list L of dictionaries D's where 
        D is of the form {'name': 'Name of a class extending FeatureExtractBase',
        'args': 'arguments (a kwargs dictionary) to class constructor'}. ?
        """
        self.features_list = features_list

    def extract(self, instance):
        feature_class_dict = {"ARFeatures":ARFeatures,
                              "FFTFeatures":FFTFeatures,
                              "RandomFeatures":RandomFeatures}
        extracted_features_list = []
        for feature_string in self.features_list:
            if feature_string['name'] in feature_class_dict:
                kwargs = feature_string['args']
                feature_object = feature_class_dict[feature_string['name']](**kwargs)
                extracted_features_list.append(np.hstack(feature_object.extract(instance))) #flattened
        return np.hstack(extracted_features_list)


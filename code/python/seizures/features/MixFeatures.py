import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.RandomFeatures import RandomFeatures

class MixFeatures(FeatureExtractBase):
    """
    Class to concatenate output of individual feature classes.
    @author V&J
    """

    def __init__(self,features_string_list):
        self.features_string_list = features_string_list

    def extract(self, instance):
    	
    	feature_class_dict = {"ARFeatures":ARFeatures,
    				  		"FFTFeatures":FFTFeatures,
    				  		"RandomFeatures":RandomFeatures}

    	extracted_features_list = []
    	for feature_string in features_string_list:
    		if feature_string in feature_class_dict:
	    		feature_object = feature_class_dict[feature_string]() 
	    		extracted_features_list.append(feature_object.extract(instance))

    	return np.dstack(extracted_features_list)


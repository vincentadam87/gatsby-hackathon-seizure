import numpy as np
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.features.ARFeatures import ARFeatures
from seizures.features.FFTFeatures import FFTFeatures
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.RandomFeatures import RandomFeatures
from seizures.features.XCHUHFeatures import XCHUHFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.features.LyapunovFeatures import LyapunovFeatures
from seizures.features.StatsFeatures import StatsFeatures

StatsFeatures
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
                              "PLVFeatures":PLVFeatures,
                              "RandomFeatures":RandomFeatures,
                              "SEFeatures":SEFeatures,
                              "LyapunovFeatures":LyapunovFeatures,
                              "StatsFeatures":StatsFeatures}

        extracted_features_list = []
        for feature_string in self.features_list:
            if feature_string['name'] in feature_class_dict:
                kwargs = feature_string['args']
                feature_object = feature_class_dict[feature_string['name']](**kwargs)
                extracted_features_list.append(np.hstack(feature_object.extract(instance))) #flattened
            else:
                print "feature not in list !!!"
        return np.hstack(extracted_features_list)

#------- end of MixFeatures ---------------

class StackFeatures(FeatureExtractBase):
    """
    A meta feature generator which stacks features generated from other
    FeatureExtractBase's.  Semantically this feature generator is the same as
    MixFeatures but directly takes in objects of subclass of
    FeatureExtractBase, unlike MixFeatures.  
    (I am just not comfortable passing class handle and bunch of arguments)

    @author Wittawat
    """

    def __init__(self, *feature_generators):
        """
        Input:
        feature_generators: a list of objects of subclass of FeatureExtractBase
        """
        self.feature_generators = feature_generators

    def extract(self, instance):
        extracted_features_list = []
        for generator in self.feature_generators:
            # a feature vector 
            assert(isinstance(generator, FeatureExtractBase))
            feature = generator.extract(instance)
            extracted_features_list.append(np.hstack(feature));
        return np.hstack(extracted_features_list)

    def __str__(self):
        subs = [str(e) for e in self.feature_generators]
        return 'Stack' + '(%s)'% (', '.join(subs))



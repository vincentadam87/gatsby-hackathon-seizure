"""
package containing classes for testing combination of features, predictor 
on specified datasets

@arthor: Wittawat
"""

from seizures.prediction.PredictorBase import PredictorBase
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.Global import Global
from seizures.data.DataLoader_v2 import DataLoader
from seizures.evaluation.XValidation import XValidation
from seizures.evaluation.performance_measures import accuracy, auc
from abc import abstractmethod
import numpy as np

class FeaturePredictorTestBase(object):
    """
    Abstract class for all feature-predictor testers.

    @author: Wittawat
    """
    def __init__(self, feature_extractor, predictor, patient, data_path):
        """
        feature_extractor: an instance of FeatureExtractBase
        predictor: an instance of PredictorBase
        patient: a string indicating a subject e.g., Dog_1
        data_path: full path to directory containing Dog_1, .. Patient_1,..
        """

        #assert(isinstance(feature_extractor, FeatureExtractBase))
        #assert(isinstance(predictor, PredictorBase))
        assert(isinstance(patient, basestring))

    @abstractmethod
    def test_combination(self, **options):
        """
        Test the predictor using features given by feature_extractor 
        on the data specified by the patient argument.
        """
        raise NotImplementedError()


class FeaturesPredictorsTestBase(object):
    """
    Abstract class for all features-predictors testers.
    The only difference to FeaturePredictorTestBase is that this class
    accepts a list of feature_extractor's and a list of predictor's.

    @author: Wittawat
    """
    def __init__(self, feature_extractors, predictors, patient, data_path):
        """
        feature_extractors: list of FeatureExtractBase's
        predictors: list of PredictorBase's
        patient: a string indicating a subject e.g., Dog_1
        data_path: full path to directory containing Dog_1, .. Patient_1,..
        """

        assert(type(feature_extractors)==type([]))
        assert(type(predictors)==type([]))
        assert(isinstance(patient, basestring))

    @abstractmethod
    def test_combination(self, **options):
        """
        Test the predictors using features given by each feature_extractor 
        on the data specified by the patient argument.
        """
        raise NotImplementedError()



class CVFeaturePredictorTester(FeaturePredictorTestBase):
    """
    An implementation of FeaturePredictorTestBase which test 
    by cross validation the given predictor using features from
    feature_extractor on the patient data.

    @author: Wittawat 
    """

    def __init__(self, feature_extractor, predictor, patient, 
            data_path=Global.path_map('clips_folder')):
        """
        feature_extractor: an instance of FeatureExtractBase
        predictor: an instance of PredictorBase
        patient: a string indicating a subject e.g., Dog_1
        """

        assert(isinstance(feature_extractor, FeatureExtractBase))
        assert(isinstance(predictor, PredictorBase))
        assert(isinstance(patient, basestring))

        self._feature_extractor = feature_extractor
        self._predictor = predictor
        self._patient = patient
        self._data_path = data_path

    @abstractmethod
    def test_combination(self, fold=3, max_segments=-1):
        """
        Test the predictor using features given by feature_extractor 
        on the data specified by the patient argument.
        Based on examples/cross_validation_test.py

        :param max_segments: maximum segments to load. -1 to use the number of 
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement. 

        return: a dictionary containing error report
        """
        predictor = self._predictor
        loader = DataLoader(self._data_path, self._feature_extractor)

        X_list,y_seizure, y_early = loader.blocks_for_Xvalidation(
                self._patient, fold, max_segments)

        # running cross validation
        #print 'Testing %d-fold CV on data of %s'%(fold, self._patient)
        #print "\ncross validation: seizures vs not"
        result_seizure = XValidation.evaluate(X_list, y_seizure, predictor, evaluation=auc)
        #print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
                #% (np.mean(result_seizure), np.std(result_seizure), result_seizure)


        #print "\ncross validation: early_vs_not"
        result_early = XValidation.evaluate(X_list, y_early, predictor, evaluation=auc)
        #print 'cross-validation results: mean = %.3f, sd = %.3f, raw scores = %s' \
                #% (np.mean(result_early), np.std(result_early), result_early)

        # dict containing bunch of reports
        r = {}
        r['predictor'] = predictor
        r['feature_extractor'] = self._feature_extractor
        # total features extracted. X_i is n x d
        r['total_features'] = X_list[0].shape[1]
        r['clips_folder'] = self._data_path
        r['patient'] = self._patient 
        r['cv_fold'] = fold
        r['seizure_mean_auc'] = np.mean(result_seizure)
        r['seizure_std_auc'] = np.std(result_seizure)
        r['early_mean_auc'] = np.mean(result_early)
        r['early_std_auc'] = np.std(result_early)
        return r



class CVFeaturesPredictorsTester(FeaturePredictorTestBase):
    """
    An implementation of FeaturesPredictorsTestBase which test 
    by cross validation the given predictors using features from each 
    feature_extractor on the patient data.

    @author: Wittawat 
    """

    def __init__(self, feature_extractors, predictors, patient, 
            data_path=Global.path_map('clips_folder')):

        assert(type(feature_extractors)==type([]))
        assert(type(predictors)==type([]))
        assert(isinstance(patient, basestring))
        self._feature_extractors = feature_extractors
        self._predictors = predictors
        self._patient = patient
        self._data_path = data_path

    @abstractmethod
    def test_combination(self, fold=3, max_segments=-1):
        """
        Test the predictors using features given by each feature_extractor 
        in feature_extractors on the data specified by the patient argument.

        :param max_segments: maximum segments to load. -1 to use the number of 
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement. 

        return: an instance of FeaturesPredictsTable         """
        L = []
        # preload data and extract features 
        features = [] # list of feature tuples. list length = len(self._feature_extractors)
        for i, feature_extractor in enumerate(self._feature_extractors):
            loader = DataLoader(self._data_path, feature_extractor)
            X_list,y_seizure, y_early = loader.blocks_for_Xvalidation(
                self._patient, fold, max_segments)
            features.append( (X_list, y_seizure, y_early) )

        # these loops can be parallelized.
        # !! Can be improved !!
        for i, feature_extractor in enumerate(self._feature_extractors):
            feature_list = []
            X_list, y_seizure, y_early = features[i]
            for j, predictor in enumerate(self._predictors):
                result_seizure = XValidation.evaluate(X_list, y_seizure, predictor, evaluation=auc)
                result_early = XValidation.evaluate(X_list, y_early, predictor, evaluation=auc)
                r = {}
                r['predictor'] = predictor
                r['feature_extractor'] = feature_extractor
                # total features extracted. X_i is n x d
                r['total_features'] = X_list[0].shape[1]
                r['clips_folder'] = self._data_path
                r['patient'] = self._patient 
                r['cv_fold'] = fold
                r['seizure_mean_auc'] = np.mean(result_seizure)
                r['seizure_std_auc'] = np.std(result_seizure)
                r['early_mean_auc'] = np.mean(result_early)
                r['early_std_auc'] = np.std(result_early)
                feature_list.append(r)
            L.append(feature_list)
        return FeaturesPredictsTable(L)

class FeaturesPredictsTable(object):
    """
    Simple class to manipuldate data table returned by
    CVFeaturesPredictorsTester. 
    
    To export the results in other ways, 
    just add your own methods here. See print_ascii_table() as an example.

    @author Wittawat
    """
    def __init__(self, features_predictors_results):
        """
        features_predictors_results: a two-dimensional list L of dictionaries
        containing error report such that L[i] is a list of reports for
        feature_extractor i on all predictors.
        """
        self.raw_table = features_predictors_results

    def print_table(self, attribute):
        
        """
        Report result in a feature_extractors x predictors table. 
        attribute specifies what entries to report in the table. 
        attribute is a string with possible values given in
        CVFeaturePredictorTester.test_combination() (the key in the returned 
        dictionary)
        """
        print "# From " + type(self).__name__.split('.')[-1]
        print "Reporting %s" % attribute
        L = self.raw_table

        from prettytable import PrettyTable
        predictors_strs = [str(rep['predictor']) for rep in L[0]]
        extractor_strs = [str(l[0]['feature_extractor']) for l in L]
        # see https://code.google.com/p/prettytable/wiki/Tutorial
        T = PrettyTable([''] + predictors_strs)
        T.padding_width = 1 # One space between column edges and contents (default)
        for i, feature_extractor in enumerate(extractor_strs):
            predictors_values = [r[attribute] for r in L[i]]
            extractor_col = extractor_strs[i]
            T.add_row([extractor_col] + predictors_values )

        print T

    def to_csv(self, file_name):
        raise NotImplementedError('Can someone implement this ? ')


    def __str__(self):
        self.print_ascii_table('seizure_mean_auc')



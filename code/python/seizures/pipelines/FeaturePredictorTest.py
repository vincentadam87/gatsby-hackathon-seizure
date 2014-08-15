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

        assert(isinstance(feature_extractor, FeatureExtractBase))
        assert(isinstance(predictor, PredictorBase))
        assert(isinstance(patient, basestring))

    @abstractmethod
    def test_combination(self, **options):
        """
        Test the predictor using features given by feature_extractor 
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
    def test_combination(self, fold=3, **options):
        """
        Test the predictor using features given by feature_extractor 
        on the data specified by the patient argument.
        Based on examples/cross_validation_test.py

        return: a dictionary containing error report
        """
        predictor = self._predictor
        loader = DataLoader(self._data_path, self._feature_extractor)

        X_list,y_seizure, y_early = loader.blocks_for_Xvalidation(
                self._patient, fold)

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




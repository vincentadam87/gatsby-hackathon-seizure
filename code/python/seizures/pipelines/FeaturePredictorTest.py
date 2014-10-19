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
from seizures.data.SubjectEEGData import SubjectEEGData
from abc import abstractmethod
from sklearn import cross_validation
from seizures.preprocessing import preprocessing
import numpy as np

from IPython.core.debugger import Tracer

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

    @staticmethod 
    def test_all_combinations(features, feature_extractors, predictors):
        """
        features is a list [(X_seizure, y_seizure, X_early, y_early)] where each element 
        in the tuple is itself a list of length = fold containing data in each 
        CV fold

        return an instance of FeaturesPredictsTable
        """
        # these loops can be parallelized.
        # !! Can be improved !!
        L = []
        for i, feature_extractor in enumerate(feature_extractors):
            feature_list = []
            X_seizure, y_seizure, X_early, y_early = features[i]
            for j, predictor in enumerate(predictors):
                print 'Evaluating feat: %s + pred: %s on seizure task'%(str(feature_extractor), str(predictor) )
                result_seizure = XValidation.evaluate(X_seizure, y_seizure, predictor, evaluation=auc)
                print 'Evaluating feat: %s + pred: %s on early seizure task'%(str(feature_extractor), str(predictor) )
                result_early = XValidation.evaluate(X_early, y_early, predictor, evaluation=auc)
                r = {}
                r['predictor'] = predictor
                r['feature_extractor'] = feature_extractor
                # total features extracted. X_i is n x d
                r['total_features'] = X_early[0].shape[1]
                r['cv_fold'] = len(X_early)
                r['seizure_mean_auc'] = np.mean(result_seizure)
                r['seizure_std_auc'] = np.std(result_seizure)
                r['early_mean_auc'] = np.mean(result_early)
                r['early_std_auc'] = np.std(result_early)
                feature_list.append(r)
            L.append(feature_list)
        return FeaturesPredictsTable(L)

    def test_combination(self, fold=3, max_segments=-1):
        """
        Test the predictors using features given by each feature_extractor 
        in feature_extractors on the data specified by the patient argument.

        :param max_segments: maximum segments to load. -1 to use the number of 
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement. 

        return: an instance of FeaturesPredictsTable         """
        # preload data and extract features 
        features = [] # list of feature tuples. list length = len(self._feature_extractors)
        for i, feature_extractor in enumerate(self._feature_extractors):
            loader = DataLoader(self._data_path, feature_extractor)
            X_list,y_seizure, y_early = loader.blocks_for_Xvalidation(
                self._patient, fold, max_segments)
            features.append( (X_list, y_seizure, X_list, y_early) )

        T = CVFeaturesPredictorsTester.test_all_combinations(features, 
                self._feature_extractors, self._predictors)
        return T


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
        T = PrettyTable(['feat. \ pred.'] + predictors_strs)
        T.padding_width = 1 # One space between column edges and contents (default)
        for i, feature_extractor in enumerate(extractor_strs):
            predictors_values = [r[attribute] for r in L[i]]
            if isinstance(predictors_values[0], float):
                predictors_values = ['%.3g'%v for v in predictors_values]
            extractor_col = extractor_strs[i]
            T.add_row([extractor_col] + predictors_values )

        print T

    def to_csv(self, file_name):
        raise NotImplementedError('Can someone implement this ? ')


    def __str__(self):
        self.print_ascii_table('seizure_mean_auc')


class CachedCVFeaPredTester(FeaturePredictorTestBase):
    """
    An implementation of FeaturesPredictorsTestBase which test 
    by cross validation the given predictors using features from each 
    feature_extractor on the patient data. Cache loaded raw data so that 
    loading is done only once. 

    @author: Wittawat 
    """

    def __init__(self, feature_extractors, predictors, patient, 
            data_path=Global.path_map('clips_folder'),params=None):

        assert(type(feature_extractors)==type([]))
        assert(type(predictors)==type([]))
        assert(isinstance(patient, basestring))
        self._feature_extractors = feature_extractors
        self._predictors = predictors
        self._patient = patient
        self._data_path = data_path
        self.params = params

    def test_combination(self, fold=3, max_segments=-1):
        """
        Test the predictors using features given by each feature_extractor 
        in feature_extractors on the data specified by the patient argument.

        :param max_segments: maximum segments to load. -1 to use the number of 
        total segments available. Otherwise, all segments (ictal and interictal)
        will be randomly subsampled without replacement. 

        return: an instance of FeaturesPredictsTable         """

        loader = SubjectEEGData(self._patient, self._data_path, use_cache=True, 
                max_train_segments=max_segments)

        # a list of (Instance, y_seizure, y_early)'s
        train_data = loader.get_train_data()
        fs = train_data[0][0].sample_rate

        # preprocessing. 
        #params = {'fs':fs,
        #  'anti_alias_cutoff': 100.,
        ##  'anti_alias_width': 30.,
        #  'anti_alias_attenuation' : 40,
        #  'elec_noise_width' :3.,
        #  'elec_noise_attenuation' : 60.0,
        #  'elec_noise_cutoff' : [49.,51.]}
        # list of preprocessed tuples
        params = self.params
        params['fs']=fs
#        for (x, y_seizure, y_early) in train_data:
#            x.eeg_data = preprocessing.preprocess_multichannel_data(x.eeg_data, self.params)

        #train_data2 = []
        #for (x, y_seizure, y_early) in train_data:
        #    x.eeg_data = preprocessing.preprocess_multichannel_data(x.eeg_data, params)
        #    train_data2.append(((x, y_seizure, y_early)))
        #train_data =train_data2

        # pre-extract features 
        features = [] # list of feature tuples. list length = len(self._feature_extractors)
        Y_seizure = np.array([y_seizure for (x, y_seizure, y_early) in train_data])
        Y_early = np.array([y_early for (x, y_seizure, y_early) in train_data])
        skf_seizure = cross_validation.StratifiedKFold(Y_seizure, n_folds=fold)
        skf_early = cross_validation.StratifiedKFold(Y_early, n_folds=fold)
        for i, feature_extractor in enumerate(self._feature_extractors):
            #print 'Extracting features with %s'%str(feature_extractor)
            #Xlist = [feature_extractor.extract(x) for  (x, y_seizure, y_early)
            #        in train_data]
            Xlist = []
            for (x, y_seizure, y_early) in train_data:
                #print '---------'
                #print x.eeg_data.shape
                params['fs']=x.sample_rate
                x.eeg_data = preprocessing.preprocess_multichannel_data(x.eeg_data, params)
                feat =feature_extractor.extract(x)
                #print x.eeg_data.shape
                #print feat.shape
                Xlist.append(feat)
            #Tracer()()
            # Xlist = list of ndarray's
            #print len(Xlist), Xlist[0].shape,len(Xlist[0])
            n = len(Xlist)
            #d = len(Xlist[0])
            d = Xlist[0].shape[0]
            # make 2d numpy array
            #print n,d
            X = np.zeros((n, d))
            #print X.shape
            for i in xrange(len(Xlist)):
                #print Xlist[i].shape, X[i, :].shape
                X[i, :] = Xlist[i].T

            # chunk data for cross validation
            # construct a list of 2d numpy arrays to be fed to XValidation
            # tr_I = train index, te_I = test index
            X_seizure = []
            y_seizure = []

            #Tracer()()
            for tr_I, te_I in skf_seizure:
                X_seizure.append(X[tr_I, :])
                y_seizure.append(Y_seizure[tr_I])
            X_early = []
            y_early = []
            for tr_I, te_I in skf_early:
                X_early.append(X[tr_I, :])
                y_early.append(Y_early[tr_I])
            features.append( (X_seizure, y_seizure, X_early, y_early) )

        T = CVFeaturesPredictorsTester.test_all_combinations(features, 
                self._feature_extractors, self._predictors)
        return T



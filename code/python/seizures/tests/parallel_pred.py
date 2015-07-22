from code.python.seizures.data.DataLoader_pred import DataLoader, make_test_label_dict

import sys

from independent_jobs.tools.Log import Log
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.tools.Log import logger
import time
import numpy as np
from seizures.evaluation.performance_measures import auc
import pickle
#from code.python.seizures.pipelines.FeaturePredictorTest import CachedCVFeaPredTester

class Parallel_job_train_test(IndependentJob):
    def __init__(self, aggregator,patient,
                 predictor_seizure,feature_extractors,data_path,sav_path):
        IndependentJob.__init__(self, aggregator)
        self.patient = patient
        self.feature_extractors = feature_extractors
        self.predictor_seizure = predictor_seizure
        self.data_path= data_path
        self.sav_path=sav_path

    def compute(self):
        logger.info("computing")
        result_list = []
        for feature_extractor in self.feature_extractors:

            print str(feature_extractor)

            # Load training data for patient
            print "Loading training data for " + self.patient

            loader = DataLoader(self.data_path, feature_extractor)
            X_train, y_seizure = loader.training_data(self.patient,max_segments=-1)
            # Train classifier
            print "Training seizure for " + self.patient
            self.predictor_seizure.fit(X_train, y_seizure)


            # Evaluate performance on test
            # Load test data
            print "Loading test data for" + self.patient

            X_test = loader.test_data(self.patient,max_segments=-1)

            fnames = loader.files
            fnames = [fname.split('/')[-1] for fname in fnames]
            d = make_test_label_dict()

            y_seizure_true = np.array([d[fname]['seizure'] for fname in fnames])

            # Perform prediction
            y_seizure_pred = self.predictor_seizure.predict(X_test)

            # Compute score
            score_seizure = auc(y_seizure_true, y_seizure_pred)



            r = {}
            r['feature_extractor'] = str(feature_extractor)
            # total features extracted. X_i is n x d
            r['seizure_auc'] = score_seizure
            r['predictor'] = 'Random Forest'
            r['filenames'] = fnames
            r['y_seizure_true'] = y_seizure_true
            r['y_seizure_pred'] = y_seizure_pred

            result_list.append(r)

        #table = FeaturesPredictsTable([feature_list])

        print self.patient
        print result_list

        pickle.dump( result_list, open( self.sav_path+"pred_result_detail_"+self.patient+".p", "wb" ) )
        result = SingleResult([])
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done computing")


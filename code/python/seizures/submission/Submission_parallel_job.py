import numpy as np
import os
from itertools import izip
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.tools.Log import logger
from seizures.data.DataLoader_slurm import DataLoader_slurm

# Define our custom Job, which inherits from base class IndependentJob

class Submission_parallel_job(IndependentJob):
    def __init__(self, aggregator,
                 feature_extractor,
                 predictor_preictal,
                 preprocess, patient, data_path):

        IndependentJob.__init__(self, aggregator)
        self.preprocess = preprocess
        self.feature_extractor = feature_extractor
        self.predictor_preictal = predictor_preictal
        self.patient = patient
        self.data_path = data_path

    def compute(self):
        logger.info("computing")
        # create ScalarResult instance

        result_lines = []
        loader = DataLoader_slurm(self.data_path,
                                  self.feature_extractor,
                                  preprocess=self.preprocess)
        # X_train is n x d
        X_train,y_preictal  = loader.training_data(self.patient)

        print 'X shape: ', X_train.shape
        print 'y_preictal shape', y_preictal.shape

        # train both models
        #print
        print "Training seizure for " + self.patient
        self.predictor_preictal.fit(X_train, y_preictal)


        pred_preictal = self.predictor_preictal.predict(X_train)
        print 'Results on training data'
        print 'preictal\tearly\tp(preictal)'
        for y_1, p_1 in izip(y_preictal, pred_preictal):
            print '%d\t%.3f' % (y_1, p_1)


        # now predict on all test points
        loader = DataLoader_slurm(self.data_path, self.feature_extractor, preprocess=self.preprocess)
        # X_test: n x d matrix
        X_test = loader.test_data(self.patient)
        test_fnames_patient = loader.files

        for ifname in range(len(test_fnames_patient)):
            fname = test_fnames_patient[ifname]
            # X is one instance
            X = X_test[ifname, :]
            # [0] to extract probability out of the ndarray
            pred_preictal = self.predictor_preictal.predict(X)[0]
            name = fname.split("/")[-1]
            result_lines.append(",".join([name, str(pred_preictal)]))

        result = SingleResult(result_lines)

        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done computing")

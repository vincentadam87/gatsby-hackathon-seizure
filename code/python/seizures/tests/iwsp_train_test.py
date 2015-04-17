'''
Created on 15 April 2015

Aim: emulate kaggle server now that we have the labels for test data
- train on kaggle_train
- test on kaggle_train


@author: vincent
'''


from independent_jobs.tools.Log import Log
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.tools.Log import logger
import time

from seizures.evaluation.performance_measures import auc

from seizures.pipelines.FeaturePredictorTest import FeaturesPredictsTable

from seizures.features.ARFeatures import *
from seizures.features.MixFeatures import StackFeatures
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.features.ARFeatures import ARFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
import numpy as np
import sys
from seizures.Global import Global

from seizures.data.DataLoader_v2 import DataLoader, make_test_label_dict

import pickle

from seizures.tests.parallel import Parallel_job

if __name__ == "__main__":

    patients = ['Dog_1','Dog_2','Dog_3','Dog_4',
                'Patient_1','Patient_2','Patient_3','Patient_4',
                'Patient_5','Patient_6','Patient_7','Patient_8']

    feature_extractors = [
        ARFeatures(),
        VarLagsARFeatures(2),
        VarLagsARFeatures(3),
        VarLagsARFeatures(4),
        VarLagsARFeatures(7),
        SEFeatures(),
        PLVFeatures(),
        StackFeatures(VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
        StackFeatures(VarLagsARFeatures(2), SEFeatures()),
        StackFeatures(ARFeatures(), VarLagsARFeatures(2), SEFeatures()),
        StackFeatures(ARFeatures(), VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
        StackFeatures(VarLagsARFeatures(2), VarLagsARFeatures(4), PLVFeatures(), SEFeatures())
    ]

    data_path = Global.path_map('clips_folder')
    predictor_seizure = ForestPredictor(n_estimators=100)
    predictor_early = ForestPredictor(n_estimators=100)

    d= make_test_label_dict()
    max_segments = -1


    sav_path = '/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/results/'

    cluster = 'on'
    type = 'slurm'

    if cluster == 'on':




        # --- Preparation ---
        Log.set_loglevel(20)
        logger.info("Start")
        # create folder name string
        foldername = Global.path_map('slurm_jobs_folder') +'/DataLoader'
        logger.info("Setting engine folder to %s" % foldername)
        # create parameter instance that is needed for any batch computation engine
        logger.info("Creating batch parameter instance")
        johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        batch_parameters = BatchClusterParameters(max_walltime=60*60*6,
                    foldername=foldername,
                    job_name_base="kaggle_iwsp_loader_"+timestr+"_",
                    parameter_prefix=johns_slurm_hack)
        # create slurm engine (which works locally)
        logger.info("Creating slurm engine instance")


        if type =='slurm':
            logger.info("Creating slurm engine instance")
            engine = SlurmComputationEngine(batch_parameters)
        elif type =='local':
            logger.info("Creating serial engine instance")
            engine = SerialComputationEngine()        # we have to collect aggregators somehow


        aggregators = []
        # submit job n times
        logger.info("Starting loop over job submission")




        # Iterate over subjects

        for patient in patients:
            print patient

            # Iterate over predictor
            result_list = []

            logger.info("Submitting job...")
            job = Parallel_job(SingleResultAggregator(),patient,predictor_early,
                     predictor_seizure,feature_extractors,data_path, sav_path)

            aggregators.append(engine.submit_job(job))

    # let the engine finish its business

    elif cluster == 'off' :
        for patient in patients:
            print patient

            # Iterate over predictor
            result_list = []
            for feature_extractor in feature_extractors:

                print str(feature_extractor)

                # Load training data for patient
                print "Loading training data for" + patient

                loader = DataLoader(data_path, feature_extractor)
                X_train, y_seizure, y_early = loader.training_data(patient,max_segments=max_segments)
                # Train classifier
                print "Training seizure for " + patient
                predictor_seizure.fit(X_train, y_seizure)
                print "Training early for " + patient
                predictor_early.fit(X_train, y_early)


                # Evaluate performance on test
                # Load test data
                print "Loading test data for" + patient

                X_test = loader.test_data(patient,max_segments=max_segments)

                fnames = loader.files
                fnames = [fname.split('/')[-1] for fname in fnames]
                d = make_test_label_dict()
                y_early_true = np.array([d[fname]['early'] for fname in fnames])
                y_seizure_true = np.array([d[fname]['seizure'] for fname in fnames])

                # Perform prediction
                y_seizure_pred = predictor_seizure.predict(X_test)
                y_early_pred = predictor_early.predict(X_test)
                # Compute score
                print len(y_early_true),len(y_early_pred)
                score_early = auc(y_early_true, y_early_pred)
                score_seizure = auc(y_seizure_true, y_seizure_pred)



                r = {}
                r['feature_extractor'] = str(feature_extractor)
                # total features extracted. X_i is n x d
                r['seizure_auc'] = score_seizure
                r['early_auc'] = score_early
                r['predictor'] = 'Random Forest'
                result_list.append(r)

            #table = FeaturesPredictsTable([feature_list])

            print patient
            print result_list

            pickle.dump( result_list, open( sav_path+"seq_result_"+patient+".p", "wb" ) )
            #table.print_table('seizure_auc')
            #table.print_table('early_auc')





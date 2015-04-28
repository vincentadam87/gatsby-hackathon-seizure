import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import os,sys
import pickle, time

sys.path.append("/nfs/nhome/live/vincenta/git/independent-jobs")



from independent_jobs.tools.Log import Log
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.tools.Log import logger

from seizures.tests.parallel import Parallel_job_Xval

from seizures.Global import Global

from seizures.features.ARFeatures import *
from seizures.features.MixFeatures import StackFeatures
from seizures.features.PLVFeatures import PLVFeatures
from seizures.features.SEFeatures import SEFeatures
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.prediction.Boosting import AdaBoostTrees
from seizures.pipelines.FeaturePredictorTest import CVFeaturesPredictorsTester
from seizures.pipelines.FeaturePredictorTest import CachedCVFeaPredTester


patients = ['Dog_1','Dog_2','Dog_3','Dog_4',
            'Patient_1','Patient_2','Patient_3','Patient_4',
            'Patient_5','Patient_6','Patient_7','Patient_8']

feature_extractors = [
    SEFeatures(),
    ARFeatures(),
    VarLagsARFeatures(2),
    VarLagsARFeatures(3),
    VarLagsARFeatures(4),
    VarLagsARFeatures(7),
    PLVFeatures(),
    StackFeatures(VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
    StackFeatures(VarLagsARFeatures(2), SEFeatures()),
    StackFeatures(ARFeatures(), VarLagsARFeatures(2), SEFeatures()),
    StackFeatures(ARFeatures(), VarLagsARFeatures(2), PLVFeatures(), SEFeatures()),
    StackFeatures(VarLagsARFeatures(2), VarLagsARFeatures(4), PLVFeatures(), SEFeatures())]

predictors = [ ForestPredictor(n_estimators=100, max_features=0.2)]

max_segments = -1


# Instruction
# Set patients to 'All' to have a single big classifier (but not very interesting)
# Set clips_folder to Global.path_map('clips_folder')+'clips_notest/' to work with all the data  (test label for all)


#clips_folder = Global.path_map('clips_folder')
clips_folder = Global.path_map('clips_folder')

sav_path = '/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/results/xval_all/'
if not os.path.exists(sav_path):
    os.makedirs(sav_path)




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
batch_parameters = BatchClusterParameters(max_walltime=60*60*10,
            foldername=foldername,
            job_name_base="kaggle_loader_"+timestr+"_",
            parameter_prefix=johns_slurm_hack)
# create slurm engine (which works locally)
logger.info("Creating slurm engine instance")
engine = SlurmComputationEngine(batch_parameters)
# we have to collect aggregators somehow
aggregators = []
# submit job n times
logger.info("Starting loop over job submission")


if type =='slurm':
    logger.info("Creating slurm engine instance")
    engine = SlurmComputationEngine(batch_parameters)
elif type =='local':
    logger.info("Creating serial engine instance")
    engine = SerialComputationEngine()        # we have to collect aggregators somehow

aggregators = []
# submit job n times
logger.info("Starting loop over job submission")


for patient in patients:


    print patient
    result_list = []
    logger.info("Submitting job...")
    job = Parallel_job_Xval(SingleResultAggregator(),feature_extractors,
                 predictors,
                 patient,
                 clips_folder,
                 max_segments,
                 sav_path)
    aggregators.append(engine.submit_job(job))


    #tester = CachedCVFeaPredTester(feature_extractors, predictors, patient, data_path=clips_folder)
    #tester = CVFeaturesPredictorsTester(feature_extractors, predictors, patient)
    # randomly select subsamples of total segments (ictal + interictal)
    #result_list = tester.test_combination(fold=10, max_segments=max_segments)

    #pickle.dump( result_list, open( sav_path+"slurm_xval_result_"+patient+".p", "wb" ) )





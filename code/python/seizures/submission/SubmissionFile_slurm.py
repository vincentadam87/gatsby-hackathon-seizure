import os
import time

from itertools import izip
from seizures.data.DataLoader_slurm import DataLoader_slurm
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.prediction.PredictorBase import PredictorBase
from seizures.Global import Global
import independent_jobs
from independent_jobs.tools.Log import Log
from independent_jobs.tools.Log import logger
from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from os.path import expanduser
from seizures.submission.Submission_parallel_job import Submission_parallel_job

class SubmissionFile_slurm():
    """
    Class to generate submission files
    
    @author Vincent and Alessandro
    """
    
    def __init__(self, data_path, patients=None):
        """
        Constructor
        
        Parameters:
        data_path   - / terminated path. This is the base folder 
        containing e.g., Dog_1/, Dog_2/, ....
        patients - a list of patient names e.g., ['Dog_1', 'Patient_2', ...]
        """
        if not os.path.isdir(data_path):
            raise ValueError('%s is not a directory.'%data_path)

        self.data_path = Global.path_map('clips_folder')

        if patients == None:
            self.patients = ["Dog_%d" % i for i in range(1, 5)] + ["Patient_%d" % i for i in range(1, 2)]
        else:
            self.patients = patients

        # will only work for single subject here...
#        self.patients = ["Patient_%d" % i for i in range(5, 9)]
#        self.patients = ["Dog_1"]
#        self.patients = ["Dog_2"]
#        self.patients = ["Patient_1"]
#        self.patients = ["Patient_4"]
    
    @staticmethod
    def get_submission_filenames():
        """
        Returns a data-frame with all filenames of the sample submission file
        """
        me = os.path.dirname(os.path.realpath(__file__))
        # data_dir = full path to data/ folder
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "sampleSubmission.csv"
        
        f = open(fname)
        lines = f.readlines()
        f.close()
        
        return [line.split(",")[0] for line in lines[1:]]
        
    @staticmethod
    def get_train_filenames():
        """
        Returns a list with names of the training data files
        """
        me = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "train_filenames.txt"
        f = open(fname)
        lines = f.readlines()
        f.close()
        return [line.rstrip('\n') for line in lines]
        
    def generate_submission(self, predictor_preictal,
                            feature_extractor,
                            output_fname="output.csv",
                            preprocess=None):
        """
        Generates a submission file for a given pair of predictors, which will
        be trained on all training data per patient/dog instance.
        
        Parameters:
        predictor_preictal - Instance of PredictorBase, fixed parameters
        feature_extractor - Instance of FeatureExtractBase, to extract test features
        output_fname      - Optional filename for result submission file
        test_filename     - Optional list of filenames to produce results on,
                            default is to use all
        """
        # make sure given objects are valid
        assert(isinstance(predictor_preictal, PredictorBase))
        assert(isinstance(feature_extractor, FeatureExtractBase))
        
        test_filenames = SubmissionFile_slurm.get_submission_filenames()
        train_filenames = SubmissionFile_slurm.get_train_filenames()
        

        # ----------------  MAP

        Log.set_loglevel(20)
        logger.info("Start")
        # create an instance of the SGE engine, with certain parameters

        # create folder name string
        foldername = Global.path_map('slurm_jobs_folder')+'/Submission'
        logger.info("Setting engine folder to %s" % foldername)

        # create parameter instance that is needed for any batch computation engine
        logger.info("Creating batch parameter instance")
        batch_parameters = BatchClusterParameters(max_walltime=3600*24, foldername=foldername)

        # create slurm engine (which works locally)
        logger.info("Creating slurm engine instance")
        engine = SlurmComputationEngine(batch_parameters)

        # we have to collect aggregators somehow
        aggregators = []

        # submit job n times
        logger.info("Starting loop over job submission")

        for patient in self.patients:
            job = Submission_parallel_job(SingleResultAggregator(),
                                          feature_extractor,
                                          predictor_preictal,
                                          preprocess,
                                          patient,
                                          self.data_path)

            aggregators.append(engine.submit_job(job))


        # let the engine finish its business
        logger.info("Wait for all call in engine")
        engine.wait_for_all()

        # the reduce part
        # lets collect the results
        results = []

        logger.info("Collecting results")

        for i in range(len(aggregators)):
            logger.info("Collecting result %d" % i)
            # let the aggregator finalize things, not really needed here but in general
            aggregators[i].finalize()

            # aggregators[i].get_final_result() returns a ScalarResult instance,
            # which we need to extract the number from

            patient_result = aggregators[i].get_final_result().result
            results.append(patient_result)

          # ----------------  REDUCE

        timestr = time.strftime("%Y%m%d-%H%M%S")
        csv_fname = 'submission_' + timestr + '_'+ output_fname + '.csv'
        my_result_folder = Global.path_map('my_result_folder')
        if not os.path.exists(my_result_folder):
            os.makedirs(my_result_folder)
        csv_path = Global.get_child_result_folder(csv_fname)
        f = open(csv_path, "w")
        print "Storing results to", csv_fname
        f.write("clip,preictal\n")

        for i in range(len(results)):
            result_lines = results[i]
            for line in result_lines[i]:
                f.write(line + '\n')
        f.close()



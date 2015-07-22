from numpy.random import randint
from time import sleep
import numpy as np
from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.ScalarResult import ScalarResult
from independent_jobs.tools.Log import logger


# Define our custom Job, which inherits from base class IndependentJob
class MyGatsbyJob(IndependentJob):
    def __init__(self, aggregator, data):
        IndependentJob.__init__(self, aggregator)
        self.data = data
    
    # we need to define the abstract compute method. It has to return an instance
    # of JobResult base class
    def compute(self):
        logger.info("computing")
        # job is to sleep for some time and return this time as an instance
        # of ScalarResult, which is a provided sub-class of JobResult
        #sleep_time = randint(10)
        
        #logger.info("sleeping for %d seconds" % sleep_time)
        #sleep(sleep_time)
        
        # create ScalarResult instance
        mean = np.mean(self.data)
        result = ScalarResult(mean)
        
        # submit the result to my own aggregator
        self.aggregator.submit_result(result)
        logger.info("done computing")
        

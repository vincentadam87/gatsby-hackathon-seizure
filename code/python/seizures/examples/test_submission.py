
import numpy as np
from seizures.submission import SubmissionFile
from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.features.MixFeatures import MixFeatures

# Example script to generate submission file


data_path =  "/nfs/data3/kaggle_seizure/scratch/Stiched_data/Dog_1/"


# Define Predictor
predictor_seizure = SVMPredictor
predictor_early = SVMPredictor

# Define Features
band_means = np.linspace(0, 200, 66)
band_width = 2
FFTFeatures_args = {'band_means':band_means, 'band_width':band_width}
feature_extractor = MixFeatures([{'name':"ARFeatures",'args':{}},
                                 {'name':"FFTFeatures",'args':FFTFeatures_args}])

submissionfile = SubmissionFile(data_path)

# Load training data
# Learn classifiers
# Make final file
submissionfile.generate_submission(predictor_seizure, predictor_early,
                            feature_extractor, output_fname="output.csv",
                            test_filenames=None):
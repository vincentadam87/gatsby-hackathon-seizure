
import numpy as np
from seizures.submission.SubmissionFile import SubmissionFile
from seizures.prediction.ForestPredictor import ForestPredictor
from seizures.features.SEFeatures import SEFeatures
from seizures.features.ARFeatures import ARFeatures
from seizures.features.PLVFeatures import PLVFeatures

from seizures.prediction.SVMPredictor import SVMPredictor
from seizures.features.MixFeatures import StackFeatures
from seizures.Global import Global

# Example script to generate submission file

#data_path =  "/nfs/data3/kaggle_seizure/scratch/Stiched_data/Dog_1/"
data_path = Global.get_subject_folder('Dog_5')
print data_path
# Define Predictor

predictor_seizure = ForestPredictor()

# Define Features
band_means = np.linspace(0, 200, 66)
band_width = 2
FFTFeatures_args = {'band_means': band_means, 'band_width': band_width}
#feature_extractor = MixFeatures([{'name': "ARFeatures", 'args': {}},
#                                 {'name': "PLVFeatures", 'args': {}}])
feature1 = ARFeatures()
feature2 = PLVFeatures()
feature_extractor = StackFeatures(feature1, feature2)


submissionfile = SubmissionFile(data_path)

# Load training data
# Learn classifiers
# Make final file
submissionfile.generate_submission(predictor_seizure, feature_extractor, output_fname="output.csv",
                            test_filenames=None)

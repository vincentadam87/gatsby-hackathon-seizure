import os
from pandas.io.parsers import read_csv


class SubmissionFile():
    @staticmethod
    def get_filename_list():
        me = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.sep.join(me.split(os.sep)[:-4]) + os.sep + "data"
        fname = data_dir + os.sep + "sampleSubmission.csv"
        print read_csv(fname)["clip"][:]
        
if __name__ == "__main__":
    
    
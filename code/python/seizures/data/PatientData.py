from abc import abstractmethod

class PatientData(object):
    """"
    A class to declare the data and all the parameters to fully characterize the data associated to a subject
    """

    subject_name = None
    root_folder = None
    data = None
    n_channel = None
    fs = None

    @abstractmethod
    def __init__(self, patientFolder):
        self.patientName = ''; # just the folder name
        pass

    @abstractmethod
    def getFilesList(self):
        pass

    @abstractmethod
    def getSamplingFrequency(self ):
        pass

    @abstractmethod
    def getSegmentNames(self):
        pass


    @abstractmethod
    def getSegment(self, name):
        pass


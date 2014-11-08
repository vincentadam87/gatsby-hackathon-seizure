from abc import abstractmethod

class SubjectDataBase(object):
    """"
    A class to declare the data and all the parameters to fully characterize the data associated to a subject
    """

    subject_name = None
    root_folder = None
    data = None
    n_channel = None
    fs = None

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def getFilesList(self):
        pass

    @abstractmethod
    def loadFile(self, filename):
        pass



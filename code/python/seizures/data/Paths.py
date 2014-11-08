class Paths(object):

    """"
    A class to declare the paths to data / code and outputs
    """

    def __init__(self, python_root, clips_folder, my_result_folder):
        self.python_root = python_root
        self.clips_folder = clips_folder
        self.my_result_folder = my_result_folder

    def setPaths(self, **kwargs):
        if 'python_root' in kwargs:
            self.python_root = kwargs['python_root']
        if 'clips_folder' in kwargs:
            self.python_root = kwargs['clips_folder']
        if 'my_result_folder' in kwargs:
            self.python_root = kwargs['my_result_folder']

    def get_python_root(self):
        return self.python_root

    def get_clips_folder(self):
        return self.clips_folder

    def get_my_result_folder(self):
        return self.my_result_folder


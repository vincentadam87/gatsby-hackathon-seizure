"""
Package containing  classes related to global configurations
"""
import os

class Global():
    def __init__(self):
        pass 

    @staticmethod
    def get_subject_folder(subject):
        """
        Return a full path to the subject. 
        Caller should terminate subject with / if needed.
        """
        clips_folder = Global.path_map('clips_folder')
        return os.path.join(clips_folder, subject)

    @staticmethod
    def get_child_result_folder(subfolder):
        """
        Return a full path to path_dict['my_result_folder']/subfolder.
        subfolder may contain many levels of subfolders.
        """
        my_result_folder = Global.path_map('my_result_folder')
        return os.path.join(my_result_folder, subfolder)

    @staticmethod
    def path_map(key):
        """
        Input: 
        key = string indicator of a path 
        Return: path identified by the key 
        """

        # this dict is read-only 
        path_dict = {}

        # full path to ..../code/python/
        path_dict['python_root'] = '/home/nuke/git/gatsby-hackathon-seizure/code/python'
        
        # full path to folder containing subject-specific folders i.e., Dog_1/,
        # Patien_1, ...
        path_dict['clips_folder'] = '/nfs/data3/kaggle_seizure/clips/'

        # full path to the result folder. This folder can be anywhere. 
        # This is mainly used for containing, for example, trained models, 
        # prediction result files, ..
        path_dict['my_result_folder'] = '/nfs/nhome/live/vincenta/Desktop/' 

        if not key in path_dict:
            raise ValueError('%s not in path_dict'%key)
        return path_dict[key]




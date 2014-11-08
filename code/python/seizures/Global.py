"""
Package containing  classes related to global configurations

@author Wittawat 
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

        # ... READ THIS ....
        # Not sure how to share configurations among collaborators.
        # Not ideal. But for now, change the following paths to your own by
        # copying the relevant lines and modify. Leave paths of other users
        # commented. Do not remove other users' paths. You don't need to commit
        # this file if there is no new key. If you define a new key, write
        # comment and you may commit it so other users can define the same key
        # pointing to his/her own path.
        #
        # Ideally configuration file should be outside version control. 
        # But anyway...

        # ---------------- Wittawat -----------------
        #full path to ..../code/python/
        path_dict['python_root'] = '/home/nuke/git/gatsby-hackathon-seizure/code/python'
        
        # full path to folder containing subject-specific folders i.e., Dog_1/,
        # Patien_1, ... To get a subject specific folder, use 
        # Global.get_subject_folder('Dog_1') for example.
        path_dict['clips_folder'] = '/home/nuke/git/gatsby-hackathon-seizure/wj_data'

        # full path to the result folder. This folder can be anywhere. 
        # This is mainly used for containing, for example, trained models, 
        # prediction result files, ..
        # To get a subfolder in this result folder, use
        # Global.get_child_result_folder('subfolder')
        path_dict['my_result_folder'] = '/home/nuke/git/gatsby-hackathon-seizure/wj_result' 
        # --------------- end Wittawat --------------

        # ---------------- Vincent -----------------
        path_dict['python_root'] = '/nfs/nhome/live/vincenta/git/gatsby-hackathon-seizure/code/python'
        path_dict['clips_folder'] = '/nfs/data3/kaggle_seizure/clips/'
        path_dict['my_result_folder'] = '/nfs/nhome/live/vincenta/Desktop/seizures_final_final/'
        # --------------- end Vincent -------------


        # ----------- don't need to modify the followings ----------
        if not key in path_dict:
            raise ValueError('%s not in path_dict'%key)
        return path_dict[key]




from os.path import expanduser


def get_data_path(data_path_fname):
    """
    Returns first line of given filename in home directory
    """
    home = expanduser("~")
    f = open(home + "/data_path.txt")
    data_path = f.readline()
    f.close()
    
    return data_path
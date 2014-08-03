from os.path import expanduser


def get_data_path():
    """
    Returns first line of data_path.txt in home directory
    """
    home = expanduser("~")
    f = open(home + "/data_path.txt")
    data_path = f.readline().strip()
    f.close()
    return data_path

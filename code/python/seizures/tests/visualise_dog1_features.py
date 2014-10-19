'''
Created on 28 Jun 2014

@author: heiko
'''
import matplotlib.pyplot as plt
import numpy as np
from seizures.data.DataLoader import DataLoader
from seizures.features.FFTFeatures import FFTFeatures
from seizures.Global import Global

def visualise():
    data_path = Global.path_map('clips_folder')

    # arbritary
    band_means = np.linspace(0, 200, 66)
    band_width = 2
    feature_extractor = FFTFeatures(band_means=band_means, band_width=band_width)
    
    loader = DataLoader(data_path, feature_extractor)
    X_list = loader.training_data("Dog_1")
    y_list = loader.labels("Dog_1")[0]
    
    plt.figure()
    for i in range(len(X_list)):
        X = X_list[i]
        
        y_seizure = y_list[i]
        
        _, _, V = np.linalg.svd(X, full_matrices=True)
        plt.plot(V[0][y_seizure == 0], V[1][y_seizure == 0], 'bo')
        plt.plot(V[0][y_seizure == 1], V[1][y_seizure == 1], 'ro')
        
    plt.show()


if __name__ == '__main__':
    visualise()

 

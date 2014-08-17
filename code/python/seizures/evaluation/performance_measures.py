import numpy as np
from sklearn import metrics


def auc(truth, prediction):
    """
    Compute the AUC for given predictions and ground truth
    
    Parameters:
    truth      - 1d numpy array of ground truth labels (binary)
    prediction - 1d numpy array of predicted probabilities
    
    @author: Heiko
    """
    
    # return integrated ROC curve
    return metrics.roc_auc_score(truth, prediction)

def accuracy(truth, prediction):
    """
    Compute the accuracy for given predictions and ground truth
    
    Parameters:
    truth      - 1d numpy array of ground truth labels (binary)
    prediction - 1d numpy array of predicted probabilities
    
    @author: Heiko
    """
    
    # return classification accuracy
    return metrics.accuracy_score(truth, prediction>.5)

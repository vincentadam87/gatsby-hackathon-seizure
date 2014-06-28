from sklearn import metrics

def auc(truth, prediction):
    """
    Compute the AUC for given predictions and ground truth
    
    Parameters:
    truth      - 1d numpy array of ground truth labels (binary)
    prediction - 1d numpy array of predicted probabilities
    """
    # compute tp & fp rates
    fpr, tpr, _ = metrics.roc_curve(truth, prediction)
    
    # return integrated ROC curve
    return metrics.auc(fpr, tpr)

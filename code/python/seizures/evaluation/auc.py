from sklearn import metrics


def auc(truth, prediction):
    """
    Compute the AUC for given predictions and ground truth
    
    Parameters:
    truth      - 1d numpy array of ground truth labels (binary)
    prediction - 1d numpy array of predicted probabilities
    
    @author: Heiko
    """
    # compute tp & fp rates, 1 is regarded as positive
    fpr, tpr, _ = metrics.roc_curve(truth, prediction, pos_label=1)
    
    # return integrated ROC curve
    return metrics.auc(fpr, tpr)

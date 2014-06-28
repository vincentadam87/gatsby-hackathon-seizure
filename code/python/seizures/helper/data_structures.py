import numpy as np


def stack_matrices(X_list):
    num_samples = np.sum([len(X) for X in X_list])
    dim = X_list[0].shape[1]
    
    X_stack = np.zeros((num_samples, dim))
    i = 0
    for i in range(len(X_list)):
        X = X_list[i]
        X_stack[i:(i + len(X))] = X
        i += len(X)
        
    return X_stack

def stack_vectors(y_list):
    num_samples = np.sum([len(y) for y in y_list])
    
    y_stack = np.zeros(num_samples)
    i = 0
    for i in range(len(y_list)):
        y = y_list[i]
        y_stack[i:(i + len(y))] = y
        i += len(y)
        
    return y_stack

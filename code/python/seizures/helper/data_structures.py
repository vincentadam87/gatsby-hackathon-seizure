import numpy as np


def stack_matrices(X_list):
    X_stack = np.vstack(X_list)
    return X_stack

def stack_vectors(y_list):
    tmp = np.concatenate(y_list)
    return tmp

def test_stack_matrices(X, X_list):
    i_begin = 0
    for X_tmp in X_list:
        i_end = i_begin + X_tmp.shape[0]
        assert np.all(X[i_begin:i_end, :] == X_tmp)
        i_begin += X_tmp.shape[0]
    print 'stack = list'


def test_stack_vectors(y, y_list):
    i_begin = 0
    for y_tmp in y_list:
        i_end = i_begin + y_tmp.shape[0]
        assert np.all(y[i_begin:i_end] == y_tmp)
        i_begin += y_tmp.shape[0]
    print 'stack = list'

# old buggy version

#def stack_matrices(X_list):
#    num_samples = np.sum([len(X) for X in X_list])
#    dim = X_list[0].shape[1]
#    
#    X_stack = np.zeros((num_samples, dim))
#    i = 0
#    for i in range(len(X_list)):
#        X = X_list[i]
#        X_stack[i:(i + len(X))] = X
#        i += len(X)
#    return X_stack

#def stack_vectors(y_list):
#    num_samples = np.sum([len(y) for y in y_list])
#    
#    y_stack = np.zeros(num_samples)
#    i = 0
#    for i in range(len(y_list)):
#        y = y_list[i]
#        y_stack[i:(i + len(y))] = y
#        i += len(y)
#        
#    return y_stack

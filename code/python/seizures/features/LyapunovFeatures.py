# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 19:10:08 2014

@author: joanasoldadomagraner
"""

import numpy as np
import scipy as sp
from seizures.features.FeatureExtractBase import FeatureExtractBase
from seizures.data.Instance import Instance
from math import log

class LyapunovFeatures(FeatureExtractBase):
    """
    Class to extract Lyapunov Exponents, taking a multivariate time series with all channels.
    @author J&V
    """

    def __init__(self):
        pass

    def extract(self, instance):
        series = instance.eeg_data.T
        n_ch, N = data.shape

         
    def d(series,a,b):
        an=series[:,a]/numpy.linalg.norm(series[:,a])
        bn=series[:,b]/numpy.linalg.norm(series[:,b])
        eps=numpy.linalg.norm(an-bn)
        return eps
    
    eps=0.1;
    dlist=[[] for i in range(N)]
    n=0 
    for i in range(N-1):
        for j in range(i+1,N-1):
            if d(series,i,j) < eps:
                n+=1
                x=min(N-i,N-j)
                if x == 5:
                    x=4
                for k in range(min(5,x)):
                    dlist[k].append(log(d(series,i+k,j+k)))
    Lyapfeat = [[] for i in range(5)]                
    for i in range(len(dlist)):
        if len(dlist[i]):
            Lyapfeat[i].append(sum(dlist[i])/len(dlist[i]))


    Lyapfeat=np.asarray(Lyapfeat)
    
    return Lyapfeat

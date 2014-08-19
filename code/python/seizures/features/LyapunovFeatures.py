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
        series = instance.eeg_data
        n_ch, N = series.shape

        def d(series,a,b):
            #print a,b,series.shape
            an=series[:,a]/np.linalg.norm(series[:,a])
            bn=series[:,b]/np.linalg.norm(series[:,b])
            eps=np.linalg.norm(an-bn)
            return eps

        eps=0.2
        dlist=[[] for i in range(5)]

        for i in range(N-1):
            for j in range(i+1,N):
                if d(series,i,j) < eps:
                    x=min(N-i,N-j)
                    for k in range(min(5,x)):
                        dlist[k].append(log(d(series,i+k,j+k)))

        #features = [[] for i in range(5)]
        features = np.zeros((5,))
        for i in range(len(dlist)):
            if len(dlist[i]):
                features[i]=np.mean(dlist[i])


        Lyapfeatures=np.asarray(features)

        return np.hstack(Lyapfeatures)

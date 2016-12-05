'''
Created on Jun 13, 2016

@author: xiul
'''

import math
import numpy as np


def initWeights(n,d):
    """ Initialization Strategy """
    #scale_factor = 0.1
    scale_factor = math.sqrt(float(6)/(n + d))
    return (np.random.rand(n,d)*2-1)*scale_factor

def mergeDicts(d0, d1):
    """ for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
    for k in d1:
        if k in d0: d0[k] += d1[k]
        else: d0[k] = d1[k]
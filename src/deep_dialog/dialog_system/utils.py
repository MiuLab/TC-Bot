"""
Created on May 25, 2016

@author: xiul, t-zalipt
"""

import numpy as np
################################################################################
#   Some helper functions
################################################################################

def unique_states(training_data):
    unique = []
    for datum in training_data:
        if contains(unique, datum[0]):
            pass
        else:
            unique.append(datum[0].copy())
    return unique

def contains(unique, candidate_state):
    for state in unique:
        if np.array_equal(state, candidate_state):
            return True
        else:
            pass
    return False

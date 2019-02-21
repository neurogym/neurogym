"""Temporary file for dependency of pyrl

TODO: remove soon
"""


from __future__ import division

from collections import OrderedDict

import numpy as np

#=========================================================================================
# Dale's principle
#=========================================================================================

def generate_ei(N, pE=0.8):
    """
    E/I signature.
    Parameters
    ----------
    N : int
        Number of recurrent units.
    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.
    """
    assert 0 <= pE <= 1

    Nexc = int(pE*N)
    Ninh = N - Nexc

    idx = range(N)
    EXC = idx[:Nexc]
    INH = idx[Nexc:]

    ei       = np.ones(N, dtype=int)
    ei[INH] *= -1

    return ei, EXC, INH

#=========================================================================================
# Functions for defining the problem
#=========================================================================================

def to_map(*args):
    if isinstance(args[0], list):
        l = args[0]
    else:
        l = args

    od = OrderedDict()
    for i, v in enumerate(l):
        od[v] = i

    return od

#=========================================================================================
# Functions for defining task epochs
#=========================================================================================

def get_idx(t, start_end):
    start, end = start_end
    return list(np.where((start <= t) & (t < end))[0])

def get_epochs_idx(dt, epochs):
    t = np.linspace(0, epochs['tmax'], int(epochs['tmax']/dt)+1)

    return t, {k: get_idx(t, v) for k, v in epochs.items() if k != 'tmax'}

#=========================================================================================
# Functions for defining datasets
#=========================================================================================

def choice(rng, a):
    return a[rng.choice(len(a))]

def unravel_index(i, dims):
    return list(np.unravel_index(i%np.prod(dims), dims, order='F'))

#=========================================================================================
# Functions for generating epoch durations that are multiples of the time step
#=========================================================================================

def uniform(rng, dt, xmin, xmax):
    return (rng.uniform(xmin, xmax)//dt)*dt

def truncated_exponential(rng, dt, mean, xmin=0, xmax=np.inf):
    while True:
        x = rng.exponential(mean)
        if xmin <= x < xmax:
            return (x//dt)*dt

#=========================================================================================
# Functions for terminating training
#=========================================================================================

def divide(x, y):
    try:
        z = x/y
        if np.isnan(z):
            raise ZeroDivisionError
        return z
    except ZeroDivisionError:
        return 0

def correct_2AFC(perf):
    p_decision = perf.n_decision/perf.n_trials
    p_correct  = divide(perf.n_correct, perf.n_decision)

    return p_decision, p_correct

#=========================================================================================
# Generic task
#=========================================================================================

class Task(object):
    def __init__(self):
        pass
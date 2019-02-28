"""Temporary file for dependency of pyrl

TODO: remove soon
"""


from __future__ import division

from collections import OrderedDict

import numpy as np


def to_map(*args):
    "produces ordered dict from given inputs"
    if isinstance(args[0], list):
        var_list = args[0]
    else:
        var_list = args
    od = OrderedDict()
    for i, v in enumerate(var_list):
        od[v] = i

    return od


def get_idx(t, start_end):
    """
    auxiliary function for defining task epochs
    """
    start, end = start_end
    return list(np.where((start <= t) & (t < end))[0])


def get_epochs_idx(dt, epochs):
    """
    function for defining task epochs
    """
    t = np.linspace(0, epochs['tmax'], int(epochs['tmax']/dt)+1)

    return t, {k: get_idx(t, v) for k, v in epochs.items() if k != 'tmax'}


def uniform(rng, dt, xmin, xmax):
    return (rng.uniform(xmin, xmax)//dt)*dt


def truncated_exponential(rng, dt, mean, xmin=0, xmax=np.inf):
    """
    function for generating epoch durations that are multiples of the time step
    """
    while True:
        x = rng.exponential(mean)
        if xmin <= x < xmax:
            return (x//dt)*dt


def choice(rng, a):
    return a[rng.choice(len(a))]


def divide(x, y):
    try:
        z = x/y
        if np.isnan(z):
            raise ZeroDivisionError
        return z
    except ZeroDivisionError:
        return 0


def correct_2AFC(perf):
    """
    computes performance
    """
    p_decision = perf.n_decision/perf.n_trials
    p_correct = divide(perf.n_correct, perf.n_decision)

    return p_decision, p_correct


def new_trial(t, tmax, dt, status, miss, reward):
    """
    check whether a new trial should be started
    """
    # new trial?
    # TODO: not negative reward for being wrong or not responding?
    new_trial = False
    if t > tmax/dt and status:
        reward = miss
        new_trial = True
    elif not status:
        new_trial = True

    return reward, new_trial


def compute_perf(perf, reward, num_tr, p_stp, num_tr_perf, tr_perf):
    if num_tr % p_stp == 0:
        print('percentage of trials performed: ' +
              str(100*num_tr_perf/p_stp))
        print('mean performnace: ' + str(perf))
        num_tr_perf = 0
        perf = 0
    elif tr_perf:
        num_tr_perf += 1
        perf += (reward - perf)/num_tr_perf

    return perf, num_tr, num_tr_perf

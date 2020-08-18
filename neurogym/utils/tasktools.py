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
    auxiliary function for defining task periods
    """
    start, end = start_end
    return list(np.where((start <= t) & (t < end))[0])


def get_periods_idx(dt, periods):
    """
    function for defining task periods
    """
    t = np.linspace(0, periods['tmax'], int(periods['tmax']/dt)+1)

    return t, {k: get_idx(t, v) for k, v in periods.items() if k != 'tmax'}


def minmax_number(dist, args):
    """Given input to the random_number_fn function, return min and max."""
    if dist == 'uniform':
        return args[0], args[1]
    elif dist == 'choice':
        return np.min(args), np.max(args)
    elif dist == 'truncated_exponential':
        return args[1], args[2]
    elif dist == 'constant':
        return args, args
    else:
        raise ValueError('Unknown dist:', str(dist))


def circular_dist(original_dist):
    '''Get the distance in periodic boundary conditions.'''
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


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


def compute_perf(perf, reward, num_tr_perf, tr_perf):
    if tr_perf:
        num_tr_perf += 1
        perf += (reward - perf)/num_tr_perf

    return perf, num_tr_perf

from collections import OrderedDict

import numpy as np


def to_map(*args):
    """Produces ordered dict from given inputs."""
    var_list = args[0] if isinstance(args[0], list) else args
    od = OrderedDict()
    for i, v in enumerate(var_list):
        od[v] = i

    return od


def get_idx(t, start_end):
    """Auxiliary function for defining task periods."""
    start, end = start_end
    return list(np.where((start <= t) & (t < end))[0])


def get_periods_idx(dt, periods):
    """Function for defining task periods."""
    t = np.linspace(0, periods["tmax"], int(periods["tmax"] / dt) + 1)

    return t, {k: get_idx(t, v) for k, v in periods.items() if k != "tmax"}


def minmax_number(dist, args):
    """Given input to the random_number_fn function, return min and max."""
    if dist == "uniform":
        return args[0], args[1]
    if dist == "choice":
        return np.min(args), np.max(args)
    if dist == "truncated_exponential":
        return args[1], args[2]
    if dist == "constant":
        return args, args
    msg = f"Unknown distribution: {dist}."
    raise ValueError(msg)


def circular_dist(original_dist):
    """Get the distance in periodic boundary conditions."""
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))


def divide(x, y):  # FIXME: why is a custom division method needed?
    try:
        z = x / y
        if np.isnan(z):
            return 0
    except ZeroDivisionError:
        return 0
    else:
        return z


def correct_2AFC(perf):  # noqa: N802
    """Computes performance."""
    p_decision = perf.n_decision / perf.n_trials
    p_correct = divide(perf.n_correct, perf.n_decision)

    return p_decision, p_correct


def compute_perf(perf, reward, num_tr_perf, tr_perf):
    if tr_perf:
        num_tr_perf += 1
        perf += (reward - perf) / num_tr_perf

    return perf, num_tr_perf

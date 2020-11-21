import numpy as np


def trunc_exp(rng, vmean, vmin=0, vmax=np.inf):
    """
    function for generating period durations
    """
    if vmin >= vmax:  # the > is to avoid issues when making vmin as big as dt
        return vmax
    else:
        while True:
            x = rng.exponential(vmean)
            if vmin <= x < vmax:
                return x


class TruncExp(object):
    def __init__(self, vmean, vmin=0, vmax=np.inf, rng=None):
        self.vmean = vmean
        self.vmin = vmin
        self.vmax = vmax
        self.rng = np.random.RandomState()

    def seed(self, seed=None):
        """Seed the PRNG of this space. """
        self.rng = np.random.RandomState(seed)

    def __call__(self, *args, **kwargs):
        if self.vmin >= self.vmax:  # the > is to avoid issues when making vmin as big as dt
            return self.vmax
        else:
            while True:
                v = self.rng.exponential(self.vmean)
                if self.vmin <= v < self.vmax:
                    return v


def random_number_fn(dist, args, rng):
    """Return a random number generating function from a distribution."""
    if dist == 'uniform':
        return lambda: rng.uniform(*args)
    elif dist == 'choice':
        return lambda: rng.choice(args)
    elif dist == 'truncated_exponential':
        return lambda: trunc_exp(rng, *args)
    elif dist == 'constant':
        return lambda: args
    else:
        raise ValueError('Unknown dist:', str(dist))


def random_number_name(dist, args):
    """Return a string explaining the dist and args."""
    if dist == 'uniform':
        return dist + ' between ' + str(args[0]) + ' and ' + str(args[1])
    elif dist == 'choice':
        return dist + ' within ' + str(args)
    elif dist == 'truncated_exponential':
        string = 'truncated exponential with mean ' + str(args[0])
        if len(args) > 1:
            string += ', min ' + str(args[1])
        if len(args) > 2:
            string += ', max ' + str(args[2])
        return string
    elif dist == 'constant':
        return dist + ' ' + str(args)
    else:
        raise ValueError('Unknown dist:', str(dist))
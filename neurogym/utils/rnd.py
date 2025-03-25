import numpy as np


def trunc_exp(rng, vmean, vmin=0, vmax=np.inf):
    """Function for generating period durations."""
    if vmin >= vmax:  # the > is to avoid issues when making vmin as big as dt
        return vmax
    while True:
        x = rng.exponential(vmean)
        if vmin <= x < vmax:
            return x


class TruncExp:
    def __init__(self, vmean, vmin=0, vmax=np.inf) -> None:
        self.vmean = vmean
        self.vmin = vmin
        self.vmax = vmax
        self.rng = np.random.RandomState()

    def seed(self, seed=None) -> None:
        """Seed the PRNG of this space."""
        self.rng = np.random.RandomState(seed)

    def __call__(self, *args, **kwargs):
        if self.vmin >= self.vmax:  # the > is to avoid issues when making vmin as big as dt
            return self.vmax
        while True:
            v = self.rng.exponential(self.vmean)
            if self.vmin <= v < self.vmax:
                return v


def random_number_fn(dist, args, rng):
    """Return a random number generating function from a distribution."""
    if dist == "uniform":
        return lambda: rng.uniform(*args)
    if dist == "choice":
        return lambda: rng.choice(args)
    if dist == "truncated_exponential":
        return lambda: trunc_exp(rng, *args)
    if dist == "constant":
        return lambda: args
    msg = f"Unknown distribution: {dist}."
    raise ValueError(msg)


def random_number_name(dist, args):
    """Return a string explaining the dist and args."""
    if dist == "uniform":
        return f"{dist} between {args[0]} and {args[1]}"
    if dist == "choice":
        return f"{dist} within {args}"
    if dist == "truncated_exponential":
        string = f"truncated exponential with mean {args[0]}"
        if len(args) > 1:
            string += f", min {args[1]}"
        if len(args) > 2:
            string += f", max {args[2]}"
        return string
    if dist == "constant":
        return f"dist{args}"
    msg = f"Unknown distribution: {dist}."
    raise ValueError(msg)

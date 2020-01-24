"""A small library for input construction."""

import numpy as np

class BaseInput(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('__call__ function not defined')


class GaussianNoise(BaseInput):
    def __init__(self, mu, sigma):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def __call__(self, shape=None):
        shape = shape or 1
        return np.random.randn(shape) * self.sigma + self.mu

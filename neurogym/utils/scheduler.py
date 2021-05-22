"""Trial scheduler class."""
import numpy as np


class BaseSchedule(object):
    """Base schedule.

    Args:
        n: int, number of conditions to schedule
    """
    def __init__(self, n):
        self.n = n
        self.total_count = 0  # total count
        self.count = 0  # count within a condition
        self.i = 0  # initialize at 0
        self.rng = np.random.RandomState()

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.total_count = 0
        self.count = 0
        self.i = 0

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SequentialSchedule(BaseSchedule):
    """Sequential schedules"""

    def __init__(self, n):
        super().__init__(n)

    def __call__(self):
        self.count = 1
        self.i += 1
        if self.i >= self.n:
            self.i = 0
        self.total_count += 1
        return self.i


class RandomSchedule(BaseSchedule):
    """Random schedules"""

    def __init__(self, n):
        super().__init__(n)

    def __call__(self):
        if self.n > 1:
            js = [j for j in range(self.n) if j != self.i]
            self.i = self.rng.choice(js)
        else:
            self.i = 0
        self.total_count += 1
        return self.i


class SequentialBlockSchedule(BaseSchedule):
    """Sequential block schedules"""

    def __init__(self, n, block_lens):
        super().__init__(n)
        self.block_lens = block_lens
        if len(block_lens) != n:
            raise ValueError('Length of block_lens must equal n')

    def __call__(self):
        if self.count < self.block_lens[self.i]:
            self.count += 1
        else:
            self.count = 1
            self.i += 1
            if self.i >= self.n:
                self.i = 0
        self.total_count += 1
        return self.i


class RandomBlockSchedule(BaseSchedule):
    """Random block schedules"""

    def __init__(self, n, block_lens):
        super().__init__(n)
        self.block_lens = block_lens
        if len(block_lens) != n:
            raise ValueError('Length of block_lens must equal n')

    def __call__(self):
        if self.count < self.block_lens[self.i]:
            self.count += 1
        else:
            self.count = 1
            if self.n > 1:
                potential_i_envs = [i for i in range(self.n) if i != self.i]
                self.i = self.rng.choice(potential_i_envs)
            else:
                self.i = 0
        self.total_count += 1
        return self.i





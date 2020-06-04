import neurogym as ngym
from neurogym.core import TrialWrapperV2
import numpy as np


class RandomGroundTruth(TrialWrapperV2):
    """"""
    def __init__(self, env, p=None):
        super().__init__(env)
        try:
            self.n_ch = len(self.choices)  # max num of choices
        except AttributeError:
            raise AttributeError('RandomGroundTruth requires task to '
                                 'have attribute choices')
        if p is None:
            p = np.ones(self.n_ch) / self.n_ch
        self.p = p

    def new_trial(self, **kwargs):
        if 'p' in kwargs:
            p = kwargs['p']
        else:
            p = self.p
        ground_truth = np.random.choice(self.env.choices, p=p)
        kwargs = {'ground_truth': ground_truth}
        return self.env.new_trial(**kwargs)


class ScheduleAttr(TrialWrapperV2):
    """Schedule attributes.

    Args:
        env: TrialEnv object
        schedule:
    """
    def __init__(self, env, schedule, attr_list):
        super().__init__(env)
        self.schedule = schedule
        self.attr_list = attr_list

    def new_trial(self, **kwargs):
        i = self.schedule()
        kwargs.update(self.attr_list[i])
        return self.env.new_trial(**kwargs)


class ScheduleEnvs(TrialWrapperV2):
    """"""
    def __init__(self, envs, schedule):
        super().__init__(envs[0])
        self.envs = envs
        self.schedule = schedule
        self.i_env = 0

    def new_trial(self, **kwargs):
        self.i_env = self.schedule()
        super().__init__(self.envs[self.i_env])
        return self.env.new_trial(**kwargs)


class TrialHistoryV2(TrialWrapperV2):
    """Change ground truth probability based on previous outcome.

    Args:
        probs: matrix of probabilities of the current choice conditioned
            on the previous. Shape, num-choices x num-choices
    """
    def __init__(self, env, probs=None):
        super().__init__(env)
        try:
            self.n_ch = len(self.choices)  # max num of choices
        except AttributeError:
            raise AttributeError('TrialHistory requires task to '
                                 'have attribute choices')
        if probs is None:
            probs = np.ones((self.n_ch, self.n_ch)) / self.n_ch  # uniform
        self.probs = probs
        assert self.probs.shape == (self.n_ch, self.n_ch), \
            'probs shape wrong, should be' + str((self.n_ch, self.n_ch))
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization

    def new_trial(self, **kwargs):
        if 'probs' in kwargs:
            probs = kwargs['probs']
        else:
            probs = self.probs
        p = probs[self.prev_trial, :]
        # Choose ground truth and update previous trial info
        self.prev_trial = self.rng.choice(self.n_ch, p=p)
        ground_truth = self.choices[self.prev_trial]
        kwargs.update({'ground_truth': ground_truth, 'probs': probs})
        return self.env.new_trial(**kwargs)

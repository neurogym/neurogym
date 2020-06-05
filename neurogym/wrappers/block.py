from gym import spaces
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
    """Schedule environments.

    Args:
        envs: list of env object
        schedule: utils.scheduler.BaseSchedule object
        env_input: bool, if True, add scalar inputs indicating current
            envinronment. default False.
    """
    def __init__(self, envs, schedule, env_input=False):
        super().__init__(envs[0])
        self.envs = envs
        self.schedule = schedule
        self.i_env = 0

        self.env_input = env_input
        if env_input:
            env_shape = envs[0].observation_space.shape
            if len(env_shape) > 1:
                raise ValueError('Env must have 1-D Box shape',
                                 'Instead got ' + str(env_shape))
            for env in envs:
                if env.observation_space.shape != env_shape:
                    raise ValueError('Env must have equal shape.')
            # self.observation_space = spaces.Box(
            #     env_shape[0] + len(self.envs)
            # )

    def new_trial(self, **kwargs):
        self.i_env = self.schedule()
        super().__init__(self.envs[self.i_env])
        if not self.env_input:
            return self.env.new_trial(**kwargs)
        else:
            self.env.new_trial(**kwargs)
            # Expand observation
            env_ob = np.zeros((self.task.ob.shape[0], len(self.envs)),
                              dtype=self.task.ob.dtype)
            env_ob[:, self.i_env] = 1.
            self.task.ob = np.concatenate((self.task.ob, env_ob), axis=-1)


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

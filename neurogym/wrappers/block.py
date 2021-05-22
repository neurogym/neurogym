from gym import spaces
import neurogym as ngym
from neurogym.core import TrialWrapper
import numpy as np


class RandomGroundTruth(TrialWrapper):
    # TODO: A better name?
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
        ground_truth = self.rng.choice(self.env.choices, p=p)
        kwargs = {'ground_truth': ground_truth}
        return self.env.new_trial(**kwargs)


class ScheduleAttr(TrialWrapper):
    """Schedule attributes.

    Args:
        env: TrialEnv object
        schedule:
    """
    def __init__(self, env, schedule, attr_list):
        super().__init__(env)
        self.schedule = schedule
        self.attr_list = attr_list

    def seed(self, seed=None):
        self.schedule.seed(seed)
        self.env.seed(seed)

    def new_trial(self, **kwargs):
        i = self.schedule()
        kwargs.update(self.attr_list[i])
        return self.env.new_trial(**kwargs)


def _have_equal_shape(envs):
    """Check if environments have equal shape."""
    env_ob_shape = envs[0].observation_space.shape
    for env in envs:
        if env.observation_space.shape != env_ob_shape:
            raise ValueError(
                'Env must have equal observation shape. Instead got' +
                str(env.observation_space.shape) + ' for ' + str(env) +
                ' and ' + str(env_ob_shape) + ' for ' + str(envs[0]))

    env_act_shape = envs[0].action_space.n
    for env in envs:
        if env.action_space.n != env_act_shape:
            raise ValueError(
                'Env must have equal action shape. Instead got ' +
                str(env.action_space.n) + ' for ' + str(env) +
                ' and ' + str(env_act_shape) + ' for ' + str(envs[0]))


class MultiEnvs(TrialWrapper):
    """Wrap multiple environments.

    Args:
        envs: list of env object
        env_input: bool, if True, add scalar inputs indicating current
            envinronment. default False.
    """
    def __init__(self, envs, env_input=False):
        super().__init__(envs[0])
        for env in envs:
            env.unwrapped.set_top(self)
        self.envs = envs
        self.i_env = 0

        self.env_input = env_input
        if env_input:
            env_shape = envs[0].observation_space.shape
            if len(env_shape) > 1:
                raise ValueError('Env must have 1-D Box shape',
                                 'Instead got ' + str(env_shape))
            _have_equal_shape(envs)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(env_shape[0] + len(self.envs),),
                dtype=self.observation_space.dtype
            )

    def set_i(self, i):
        """Set the i-th environment."""
        self.i_env = i
        self.env = self.envs[self.i_env]

    def new_trial(self, **kwargs):
        if not self.env_input:
            return self.env.new_trial(**kwargs)
        else:
            trial = self.env.new_trial(**kwargs)
            # Expand observation
            env_ob = np.zeros((self.unwrapped.ob.shape[0], len(self.envs)),
                              dtype=self.unwrapped.ob.dtype)
            env_ob[:, self.i_env] = 1.
            self.unwrapped.ob = np.concatenate(
                (self.unwrapped.ob, env_ob), axis=-1)
            return trial


class ScheduleEnvs(TrialWrapper):
    """Schedule environments.

    Args:
        envs: list of env object
        schedule: utils.scheduler.BaseSchedule object
        env_input: bool, if True, add scalar inputs indicating current
            envinronment. default False.
    """
    def __init__(self, envs, schedule, env_input=False):
        super().__init__(envs[0])
        for env in envs:
            env.unwrapped.set_top(self)
        self.envs = envs
        self.schedule = schedule
        self.i_env = 0

        self.env_input = env_input
        if env_input:
            env_shape = envs[0].observation_space.shape
            if len(env_shape) > 1:
                raise ValueError('Env must have 1-D Box shape',
                                 'Instead got ' + str(env_shape))
            _have_equal_shape(envs)
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(env_shape[0] + len(self.envs),),
                dtype=self.observation_space.dtype
            )

    def seed(self, seed=None):
        for env in self.envs:
            env.seed(seed)
        self.schedule.seed(seed)

    def new_trial(self, **kwargs):
        self.i_env = self.schedule()
        self.env = self.envs[self.i_env]
        if not self.env_input:
            return self.env.new_trial(**kwargs)
        else:
            trial = self.env.new_trial(**kwargs)
            # Expand observation
            env_ob = np.zeros((self.unwrapped.ob.shape[0], len(self.envs)),
                              dtype=self.unwrapped.ob.dtype)
            env_ob[:, self.i_env] = 1.
            self.unwrapped.ob = np.concatenate(
                (self.unwrapped.ob, env_ob), axis=-1)
            return trial


class TrialHistoryV2(TrialWrapper):
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

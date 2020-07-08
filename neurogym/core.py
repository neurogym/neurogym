#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import gym
import warnings

from neurogym.utils import tasktools

METADATA_DEF_KEYS = ['description', 'paper_name', 'paper_link', 'timing',
                     'tags']


def _clean_string(string):
    return ' '.join(string.replace('\n', '').split())


def env_string(env):
    string = ''
    metadata = env.metadata
    docstring = env.__doc__
    string += "### {:s}\n".format(type(env).__name__)
    paper_name = metadata.get('paper_name',
                              None) or 'Missing paper name'
    paper_name = _clean_string(paper_name)
    paper_link = metadata.get('paper_link', None)
    string += "Doc: {:s}\n".format(docstring)
    string += "Reference paper \n"
    if paper_link is None:
        string += "{:s}\n".format(paper_name)
        string += 'Missing paper link\n'
    else:
        string += "[{:s}]({:s})\n".format(paper_name, paper_link)
    # add timing info
    if isinstance(env, TrialEnv):
        timing = env.timing
        string += '\nPeriod timing (ms) \n'
        for key, val in timing.items():
            dist, args = val
            string +=\
                key + ' : ' + tasktools.random_number_name(dist, args) + '\n'

    if env.rewards:
        string += '\nReward structure \n'
        for key, val in env.rewards.items():
            string += key + ' : ' + str(val) + '\n'

    # add extra info
    other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
    if len(other_info) > 0:
        string += "\nOther parameters: \n"
        for key in other_info:
            string += key + ' : ' + _clean_string(str(metadata[key])) + '\n'
    # tags
    tags = metadata['tags']
    string += '\nTags: '
    for tag in tags:
        string += tag + ', '
    string = string[:-2] + '.\n'
    return string


class BaseEnv(gym.Env):
    """The base Neurogym class to include dt"""

    def __init__(self, dt=100):
        super(BaseEnv, self).__init__()
        self.dt = dt
        self.seed()

    # Auxiliary functions
    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        """Do nothing. Run one step"""
        return self.step(self.action_space.sample())


class TrialEnv(BaseEnv):
    """The main Neurogym class for trial-based envs."""

    def __init__(self, dt=100, num_trials_before_reset=10000000, r_tmax=0):
        super(TrialEnv, self).__init__(dt=dt)
        self.dt = dt
        self.t = self.t_ind = 0
        self.tmax = 10000  # maximum time steps
        self.r_tmax = r_tmax
        self.num_tr = 0
        self.num_tr_exp = num_trials_before_reset
        self.trial = None
        self._trial_built = False

        self.performance = 0
        # Annotations of observation space and action space
        self.ob_dict = {}
        self.act_dict = {}
        self.rewards = {}
        self._default_ob_value = False  # default to have no specific value

        # For optional periods
        self.timing = {}
        self.start_t = dict()
        self.end_t = dict()
        self.start_ind = dict()
        self.end_ind = dict()
        self._tmax = 0  # Length of each trial

        self._top = self

        self.seed()

    def __str__(self):
        """Information about task."""
        return env_string(self)  # TODO: simplify, too long now

    def _new_trial(self, **kwargs):
        """Private interface for starting a new trial.

        This function can typically update the self.trial
        dictionary that contains information about current trial
        TODO: Need to clearly define the expected behavior

        Args:

        Returns:
            observation: numpy array of agent's observation during the trial
            target: numpy array of target action of the agent
            trial: dict of trial information. Available to step function as
                self.trial
        """
        raise NotImplementedError('new_trial is not defined by user.')

    def _step(self, action):
        """Private interface for the environment.

        Receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        raise NotImplementedError('_step is not defined by user.')

    def new_trial(self, **kwargs):
        """Public interface for starting a new trial.

        Args:

        Returns:
            observation: numpy array of agent's observation during the trial
            target: numpy array of target action of the agent
            trial: dict of trial information. Available to step function as
                self.trial
        """
        self._new_trial(**kwargs)
        self.num_tr += 1  # Increment trial count
        self._trial_built = False

    def step(self, action):
        """Public interface for the environment."""
        obs, reward, done, info = self._step(action)

        self.t += self.dt  # increment within trial time count
        self.t_ind += 1

        if self.t > self.tmax - self.dt and not info['new_trial']:
            info['new_trial'] = True
            reward += self.r_tmax

        # TODO: Handle the case when new_trial is not provided in info
        # TODO: new_trial happens after step, so trial indx precedes obs change
        if info['new_trial']:
            info['performance'] = self.performance
            self.performance = 0
            self.t = self.t_ind = 0  # Reset within trial time count
            self._top.new_trial()
            if self.trial:
                info.update(self.trial)
        return obs, reward, done, info

    def reset(self, step_fn=None, no_step=False):
        """Reset the environment.

        Args:
            new_tr_fn: function or None. If function, overwrite original
                self.new_trial function
            step_fn: function or None. If function, overwrite original
                self.step function
            no_step: bool. If True, no step is taken and observation randomly
                sampled. Default False.
        """
        # TODO: Why are we stepping during reset?
        self.num_tr = 0
        self.t = self.t_ind = 0

        # TODO: Check this works with wrapper
        self._top.new_trial()
        self.action_space.seed(0)
        if no_step:
            return self.observation_space.sample()
        if step_fn is None:
            # obs, _, _, _ = self.step(self.action_space.sample())
            obs, _, _, _ = self._top.step(self.action_space.sample())
        else:
            obs, _, _, _ = step_fn(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass

    def set_top(self, wrapper):
        """Set top to be wrapper."""
        self._top = wrapper

    def sample_time(self, period):
        timing = self.timing[period]
        if isinstance(timing, (int, float)):
            t = timing
        elif callable(timing):
            t = timing()
        elif isinstance(timing[0], (int, float)):
            # Expect list of int/float, and use random choice
            t = self.rng.choice(timing)
        else:
            dist, args = timing
            if dist == 'uniform':
                t = self.rng.uniform(*args)
            elif dist == 'choice':
                t = self.rng.choice(args)
            elif dist == 'truncated_exponential':
                t = tasktools.trunc_exp_new(self.rng, *args)
            elif dist == 'constant':
                t = args
            else:
                raise ValueError('Unknown dist:', str(dist))
        return (t // self.dt) * self.dt

    def add_period(self, period, duration=None, before=None, after=None,
                   last_period=False):
        """Add an period.

        Args:
            period: string or list of strings, name of the period
            duration: float or None, duration of the period
                if None, inferred from timing_fn
            before: (optional) str, name of period that this period is before
            after: (optional) str, name of period that this period is after
                or float, time of period start
            last_period: bool, default False. If True, then this is last period
                will generate self.tmax, self.tind, and self.ob
        """
        assert not self._trial_built, 'Cannot add period after trial ' \
                                      'is built, i.e. after running add_ob'
        if isinstance(period, str):
            pass
        else:
            if duration is None:
                duration = [None] * len(period)
            else:
                assert len(duration) == len(period),\
                    'duration and period must have same length'

            # Recursively calling itself
            self.add_period(period[0], duration=duration[0], after=after)
            for i in range(1, len(period)):
                is_last = (i == len(period) - 1) and last_period
                self.add_period(period[i], duration=duration[i],
                                after=period[i - 1], last_period=is_last)
            return

        if duration is None:
            # duration = (self.timing_fn[period]() // self.dt) * self.dt
            duration = self.sample_time(period)

        if after is not None:
            if isinstance(after, str):
                start = self.end_t[after]
            else:
                start = after
        elif before is not None:
            start = self.start_t[before] - duration
        else:
            start = 0  # default start with 0

        self.start_t[period] = start
        self.end_t[period] = start + duration
        self.start_ind[period] = int(start/self.dt)
        self.end_ind[period] = int((start + duration)/self.dt)

        self._tmax = max(self._tmax, start + duration)

    def _init_trial(self):
        """Initialize trial info with tmax, tind, ob"""
        tmax_ind = int(self._tmax/self.dt)
        self.tmax = tmax_ind * self.dt
        ob_shape = [tmax_ind] + list(self.observation_space.shape)
        if self._default_ob_value:
            self.ob = np.full(ob_shape, self._default_ob_value,
                              dtype=self.observation_space.dtype)
        else:
            self.ob = np.zeros(ob_shape, dtype=self.observation_space.dtype)
        self.gt = np.zeros([tmax_ind] + list(self.action_space.shape),
                           dtype=self.action_space.dtype)
        self._trial_built = True

    def view_ob(self, period=None):
        """View observation of an period."""
        if not self._trial_built:
            self._init_trial()

        if period is None:
            return self.ob
        else:
            return self.ob[self.start_ind[period]:self.end_ind[period]]

    def _add_ob(self, value, period=None, where=None, reset=False):
        """Set observation in period to value.

        Args:
            value: np array (ob_space.shape, ...)
            period: string, must be name of an added period
            where: string or np array, location of stimulus to be added
        """
        if isinstance(period, str) or period is None:
            pass
        else:
            for p in period:
                self._add_ob(value, p, where, reset=reset)
            return

        # self.ob[self.start_ind[period]:self.end_ind[period]] = value
        ob = self.view_ob(period=period)
        if where is None:
            if reset:
                ob *= 0
            try:
                ob += value(ob)
            except TypeError:
                ob += value
        else:
            if isinstance(where, str):
                where = self.ob_dict[where]
            # TODO: This only works if the slicing is one one-dimension
            if reset:
                ob[..., where] *= 0
            try:
                ob[..., where] += value(ob[..., where])
            except TypeError:
                ob[..., where] += value

    def add_ob(self, value, period=None, where=None):
        """Add value to observation.

        Args:
            period: string, must be name of an added period
            value: np array (ob_space.shape, ...)
            where: string or np array, location of stimulus to be added
        """
        self._add_ob(value, period, where, reset=False)

    def add_randn(self, mu=0, sigma=1, period=None, where=None):
        if isinstance(period, str) or period is None:
            pass
        else:
            for p in period:
                self.add_randn(mu, sigma, p, where)
            return

        ob = self.view_ob(period=period)
        if where is None:
            ob += mu + self.rng.randn(*ob.shape) * sigma
        else:
            if isinstance(where, str):
                where = self.ob_dict[where]
            # TODO: This only works if the slicing is one one-dimension
            ob[..., where] += mu + self.rng.randn(*ob[..., where].shape) * sigma

    def set_ob(self, value, period=None, where=None):
        self._add_ob(value, period, where, reset=True)

    def set_groundtruth(self, value, period):
        """Set groundtruth value."""
        if isinstance(period, str):
            self.gt[self.start_ind[period]: self.end_ind[period]] = value
        else:
            for p in period:
                self.set_groundtruth(value, p)

    def view_groundtruth(self, period):
        """View observation of an period."""
        return self.gt[self.start_ind[period]:self.end_ind[period]]

    def in_period(self, period, t=None):
        """Check if current time or time t is in period"""
        if t is None:
            t = self.t  # Default
        return self.start_t[period] <= t < self.end_t[period]

    @property
    def ob_now(self):
        return self.ob[self.t_ind]

    @property
    def gt_now(self):
        return self.gt[self.t_ind]


# TODO: How to prevent the repeated typing here?
class TrialWrapper(gym.Wrapper):
    """Base class for wrapping TrialEnv"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task = self.unwrapped

    def new_trial(self, **kwargs):
        self.env.new_trial()

    def reset(self, step_fn=None):
        """
        restarts the experiment with the same parameters
        """
        stp_fn = step_fn or self.step
        obs = self.env.reset(step_fn=stp_fn)
        return obs

    def step(self, action):
        """Public interface for the environment."""
        # TODO: Relying on private interface will break some gym behavior
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


# TODO: How to prevent the repeated typing here?
class TrialWrapperV2(gym.Wrapper):
    """Base class for wrapping TrialEnv"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        if not isinstance(self.unwrapped, TrialEnv):
            raise TypeError("Trial wrapper must be used on TrialEnv"
                            "Got instead", self.unwrapped)
        self.unwrapped.set_top(self)

    @property
    def task(self):
        """Alias."""
        return self.unwrapped

    def new_trial(self, **kwargs):
        raise NotImplementedError

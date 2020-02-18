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
    docstring = env.__init__.__doc__
    string += "### {:s}\n".format(type(env).__name__)
    paper_name = metadata.get('paper_name',
                              None) or 'Missing paper name'
    paper_name = _clean_string(paper_name)
    paper_link = metadata.get('paper_link', None)
    string += "Doc: {:s}\n".format(docstring)
    string += "Reference paper: \n"
    if paper_link is None:
        string += "{:s}\n".format(paper_name)
        string += 'Missing paper link\n'
    else:
        string += "[{:s}]({:s})\n".format(paper_name, paper_link)
    # add timing info
    if isinstance(env, PeriodEnv):
        timing = metadata['timing']
        string += '\nDefault Period timing (ms) \n'
        for key, val in timing.items():
            dist, args = val
            string += key + ' : ' + dist + ' ' + str(args) + '\n'
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
        self.rng = np.random
        self.rng.seed(seed)
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
        self.performance = 0
        # flag indicating whether the trial was succesfully built
        self.trial_built = False
        # Annotations of observation space and action space
        self.ob_dict = {}
        self.act_dict = {}
        self.rewards = {}

        self.seed()

    def new_trial(self, **kwargs):
        """Public interface for starting a new trial.

        This function can typically update the self.trial
        dictionary that contains information about current trial
        TODO: Need to clearly define the expected behavior
        """
        raise NotImplementedError('new_trial is not defined by user.')

    def _step(self, action):
        """Private interface for the environment.

        Receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        raise NotImplementedError('_step is not defined by user.')

    def step(self, action):
        """Public interface for the environment."""
        obs, reward, done, info = self._step(action)

        self.t += self.dt  # increment within trial time count
        self.t_ind += 1

        if self.t > self.tmax - self.dt and not info['new_trial']:
            info['new_trial'] = True
            info['trial_endwith_tmax'] = True  # TODO: do whe need this?
            reward += self.r_tmax
        else:
            info['trial_endwith_tmax'] = False

        # TODO: Handle the case when new_trial is not provided in info
        # TODO: new_trial happens after step, so trial indx precedes obs change
        if info['new_trial']:
            info['performance'] = self.performance
            self.performance = 0
            self.t = self.t_ind = 0  # Reset within trial time count
            self.num_tr += 1  # Increment trial count
            self.trial_built = False
            self.new_trial(info=info)
            assert self.trial_built, 'Trial was not succesfully built.' +\
                ' (Hint: make last_period=True when adding the last period)'
        return obs, reward, done, info

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        self.num_tr = 0
        self.t = self.t_ind = 0

        # TODO: Check this works with wrapper
        self.new_trial()
        obs, _, _, _ = self.step(0)
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass


class PeriodEnv(TrialEnv):
    """Environment class with trial/period structure."""

    def __init__(self, dt=100, timing=None, num_trials_before_reset=10000000,
                 r_tmax=0):
        super(PeriodEnv, self).__init__(
            dt=dt, num_trials_before_reset=num_trials_before_reset,
            r_tmax=r_tmax)

        self.gt = None

        default_timing = self.metadata['timing'].copy()
        if timing is not None:
            default_timing.update(timing)
        self.timing = default_timing
        self.timing_fn = dict()
        for key, val in self.timing.items():
            dist, args = val
            self.timing_fn[key] = tasktools.random_number_fn(dist, args,
                                                             self.rng)
            min_tmp, max_tmp = tasktools.minmax_number(dist, args)
            if min_tmp < self.dt:
                warnings.warn('Warning: Minimum time for period {:s} {:f} smaller than dt {:f}'.format(
                    key, min_tmp, self.dt))

        self.start_t = dict()
        self.end_t = dict()
        self.start_ind = dict()
        self.end_ind = dict()

    def __str__(self):
        """Information about task."""
        return env_string(self)

    def add_period(self, period, duration=None, before=None, after=None,
                   last_period=False):
        """Add an period.

        Args:
            period: string, name of the period
            duration: float or None, duration of the period
                if None, inferred from timing_fn
            before: (optional) str, name of period that this period is before
            after: (optional) str, name of period that this period is after
                or float, time of period start
            last_period: bool, default False. If True, then this is last period
                will generate self.tmax, self.tind, and self.obs
        """
        if duration is None:
            duration = (self.timing_fn[period]() // self.dt) * self.dt
        if duration == self.dt:
            warnings.warn('Warning: Time for period {:s} {:f}'.format(period,
                                                                      duration,
                                                                      self.dt)
                          + '  lasts only one timestep. Agents will not have' +
                          ' time to respond (e.g. make a choice) on time.')

        if after is not None:
            if isinstance(after, str):
                start = self.end_t[after]
            else:
                start = after
        elif before is not None:
            start = self.start_t[before] - duration
        else:
            raise ValueError('''before or start can not be both None''')

        self.start_t[period] = start
        self.end_t[period] = start + duration
        self.start_ind[period] = int(start/self.dt)
        self.end_ind[period] = int((start + duration)/self.dt)

        if last_period:
            self.trial_built = True
            self._init_trial(start + duration)

    def _init_trial(self, tmax):
        """Initialize trial info with tmax, tind, obs"""
        tmax_ind = int(tmax/self.dt)
        self.tmax = tmax_ind * self.dt
        self.obs = np.zeros([tmax_ind] + list(self.observation_space.shape))
        self.gt = np.zeros([tmax_ind] + list(self.action_space.shape))

    def view_ob(self, period=None):
        """View observation of an period."""
        if period is None:
            return self.obs
        else:
            return self.obs[self.start_ind[period]:self.end_ind[period]]

    def _add_ob(self, period, value, where=None, reset=False):
        """Set observation in period to value.

        Args:
            period: string, must be name of an added period
            value: np array (ob_space.shape, ...)
            where: string or np array, location of stimulus to be added
        """
        # self.obs[self.start_ind[period]:self.end_ind[period]] = value
        ob = self.view_ob(period=period)
        if reset:
            ob *= 0
        if where is None:
            try:
                ob += value(ob)
            except TypeError:
                ob += value
        else:
            if isinstance(where, str):
                where = self.ob_dict[where]
            # TODO: This only works if the slicing is one one-dimension
            try:
                ob[..., where] += value(ob[..., where])
            except TypeError:
                ob[..., where] += value

    def add_ob(self, period, value, where=None):
        """Add value to observation.

        Args:
            period: string, must be name of an added period
            value: np array (ob_space.shape, ...)
            where: string or np array, location of stimulus to be added
        """
        self._add_ob(period, value, where, reset=False)

    def set_ob(self, period, value, where=None):
        self._add_ob(period, value, where, reset=True)

    def set_groundtruth(self, period, value):
        """Set groundtruth value."""
        self.gt[self.start_ind[period]: self.end_ind[period]] = value

    def view_groundtruth(self, period):
        """View observation of an period."""
        return self.gt[self.start_ind[period]:self.end_ind[period]]

    def in_period(self, period, t=None):
        """Check if current time or time t is in period"""
        if t is None:
            t = self.t  # Default
        return self.start_t[period] <= t < self.end_t[period]

    @property
    def obs_now(self):
        return self.obs[self.t_ind]

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
        raise NotImplementedError('_new_trial need to be implemented')

    def step(self, action):
        """Public interface for the environment."""
        # TODO: Relying on private interface will break some gym behavior
        # TODO: Manually updating task.t here is bad shouldn't allow other
        # things to change it
        obs, reward, done, info = self.task._step(action)
        self.task.t += self.task.dt  # increment within trial time count
        self.task.t_ind += 1

        if self.t > self.tmax - self.dt:
            info['new_trial'] = True

        if info['new_trial']:
            info['performance'] = self.performance
            self.performance = 0
            self.task.t = self.task.t_ind = 0  # Reset within trial time count
            self.task.num_tr += 1  # Increment trial count
            self.new_trial(info=info)  # new_trial from wrapper
        return obs, reward, done, info

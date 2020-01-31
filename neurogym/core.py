#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import gym
import warnings

from neurogym.ops import tasktools


class BaseEnv(gym.Env):
    """The base Neurogym class to include dt"""

    def __init__(self, dt=100):
        super(BaseEnv, self).__init__()
        self.dt = dt
        self.seed()

    # Auxiliary functions
    def seed(self, seed=None):
        self.rng = random
        self.rng.seed(seed)
        return [seed]

    def reset(self):
        """Do nothing. Run one step"""
        return self.step(self.action_space.sample())


class TrialEnv(BaseEnv):
    """The main Neurogym class for trial-based tasks."""

    def __init__(self, dt=100, num_trials_before_reset=10000000, r_tmax=0):
        super(TrialEnv, self).__init__(dt=dt)
        self.dt = dt
        self.t = self.t_ind = 0
        self.tmax = 10000  # maximum time steps
        self.r_tmax = r_tmax
        self.num_tr = 0
        self.num_tr_exp = num_trials_before_reset
        self.trial = None
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
        # TODO: new_trial happens after step, so trial index precedes obs change
        if info['new_trial']:
            self.t = self.t_ind = 0  # Reset within trial time count
            self.num_tr += 1  # Increment trial count
            self.new_trial(info=info)
        return obs, reward, done, info

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        self.num_tr = 0
        self.t = self.t_ind = 0

        # TODO: Check this works with wrapper
        self.new_trial()
        obs, _, _, _ = self.step(self.action_space.sample())
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
        self._timing = default_timing
        self.timing_fn = dict()
        for key, val in self._timing.items():
            dist, args = val
            self.timing_fn[key] = tasktools.random_number_fn(dist, args)
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
        total_min, total_max = 0, 0  # min and max time length of trial
        string = ''
        for key, val in self._timing.items():
            dist, args = val
            string += 'Period ' + key + '\n'
            string += '    ' + dist + ' ' + str(args) + '\n'

            min_tmp, max_tmp = tasktools.minmax_number(dist, args)
            total_min += min_tmp
            total_max += max_tmp  # XXX: is there a fn that provides total_max?

        string += 'Time step {:0.2f}ms\n'.format(self.dt)
        string += 'Estimate time per trial assuming sequential period\n'
        string += 'Min/Max: {:0.2f}/{:0.2f}\n'.format(total_min, total_max)
        return string

    def add_period(self, period, duration=None, before=None, after=None,
                  last_period=False):
        """Add an period.

        Args:
            period: string, name of the period
            duration: float or None, duration of the period
                if None, inferred from timing_fn
            before: (optional) string, name of period that this period is before
            after: (optional) string, name of period that this period is after
                or float, time of period start
            last_period: bool, default False. If True, then this is last period
                will generate self.tmax, self.tind, and self.obs
        """
        if duration is None:
            duration = (self.timing_fn[period]() // self.dt) * self.dt

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
            self._init_trial(start + duration)

    def _init_trial(self, tmax):
        """Initialize trial info with tmax, tind, obs"""
        tmax_ind = int(tmax/self.dt)
        self.tmax = tmax_ind * self.dt
        self.obs = np.zeros([tmax_ind] + list(self.observation_space.shape))
        self.gt = np.zeros([tmax_ind] + list(self.action_space.shape))

    def add_input(self, input, loc=None, period=None):
        """Add an input to current observation."""
        if isinstance(period, str):
            self._add_input(input, loc, period)
        else:
            for e in period:
                self._add_input(input, loc, e)

    def _add_input(self, input, loc=None, period=None):
        """Add an input to current observation."""
        if period is None:
            ob = self.obs
        else:
            ob = self.view_ob(period)

        if loc is None:
            try:
                ob[:, :] += input()
            except TypeError:
                ob[:, :] += input
        else:
            try:
                ob[:, loc] += input()
            except TypeError:
                ob[:, loc] += input

    def set_ob(self, period, value):
        """Set observation in period to value.

        Args:
            period: string, must be name of an added period
            value: np array (ob_space.shape, ...)
        """
        self.obs[self.start_ind[period]:self.end_ind[period]] = value

    def view_ob(self, period):
        """View observation of an period."""
        return self.obs[self.start_ind[period]:self.end_ind[period]]

    def add_ob(self, period, value):
        """Add value to observation.

        Args:
            period: string, must be name of an added period
            value: np array (ob_space.shape, ...)
        """
        ob = self.view_ob(period)
        ob += value  # in-place

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
            self.task.t = self.task.t_ind = 0  # Reset within trial time count
            self.task.num_tr += 1  # Increment trial count
            self.new_trial(info=info)  # new_trial from wrapper
        return obs, reward, done, info

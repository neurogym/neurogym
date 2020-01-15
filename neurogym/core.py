#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import gym

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

    def __init__(self, dt=100, num_trials_before_reset=10000000):
        super(TrialEnv, self).__init__(dt=dt)
        self.dt = dt
        self.t = self.t_ind = 0
        self.tmax = 10000  # maximum time steps
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

        if self.t >= self.tmax - self.dt:
            info['new_trial'] = True

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


class EpochEnv(TrialEnv):
    """Environment class with trial/epoch structure."""

    def __init__(self, dt=100, timing=None, num_trials_before_reset=10000000):
        super(EpochEnv, self).__init__(
            dt=dt, num_trials_before_reset=num_trials_before_reset)

        self.gt = None

        default_timing = self.metadata['default_timing'].copy()
        if timing is not None:
            default_timing.update(timing)
        self._timing = default_timing
        self.timing_fn = dict()
        for key, val in self._timing.items():
            dist, args = val
            self.timing_fn[key] = tasktools.random_number_fn(dist, args)
            min_tmp, max_tmp = tasktools.minmax_number(dist, args)
            if min_tmp < self.dt:
                print('Warning: Minimum time for epoch {:s} {:f} smaller than dt {:f}'.format(
                    key, min_tmp, self.dt))

        self._start_t = dict()
        self._end_t = dict()
        self._start_ind = dict()
        self._end_ind = dict()

    def __str__(self):
        """Information about task."""
        total_min, total_max = 0, 0  # min and max time length of trial
        string = ''
        for key, val in self._timing.items():
            dist, args = val
            string += 'Epoch ' + key + '\n'
            string += '    ' + dist + ' ' + str(args) + '\n'

            min_tmp, max_tmp = tasktools.minmax_number(dist, args)
            total_min += min_tmp
            total_max += max_tmp  # XXX: is there a fn that provides total_max?

        string += 'Time step {:0.2f}ms\n'.format(self.dt)
        string += 'Estimate time per trial assuming sequential epoch\n'
        string += 'Min/Max: {:0.2f}/{:0.2f}\n'.format(total_min, total_max)
        return string

    def add_epoch(self, epoch, duration=None, before=None, after=None,
                  last_epoch=False):
        """Add an epoch.

        Args:
            epoch: string, name of the epoch
            duration: float or None, duration of the epoch
                if None, inferred from timing_fn
            before: (optional) string, name of epoch that this epoch is before
            after: (optional) string, name of epoch that this epoch is after
                or float, time of epoch start
            last_epoch: bool, default False. If True, then this is last epoch
                will generate self.tmax, self.tind, and self.obs
        """
        if duration is None:
            duration = (self.timing_fn[epoch]() // self.dt) * self.dt

        if after is not None:
            if isinstance(after, str):
                start = self._end_t[after]
            else:
                start = after
        elif before is not None:
            start = self._start_t[before] - duration
        else:
            raise ValueError('''before or start can not be both None''')

        self._start_t[epoch] = start
        self._end_t[epoch] = start + duration
        self._start_ind[epoch] = int(start/self.dt)
        self._end_ind[epoch] = int((start + duration)/self.dt)

        if last_epoch:
            self._init_trial(start + duration)

    def _init_trial(self, tmax):
        """Initialize trial info with tmax, tind, obs"""
        tmax_ind = int(tmax/self.dt)
        self.tmax = tmax_ind * self.dt
        ob_shape = [tmax_ind] + list(self.observation_space.shape)
        self.obs = np.zeros(ob_shape)

        # TODO: Allow ground truth to be category or full action
        self.gt = np.zeros(tmax_ind, dtype=np.int)  # ground truth action, default 0

    def set_ob(self, epoch, value):
        """Set observation in epoch to value.

        Args:
            epoch: string, must be name of an added epoch
            value: np array (ob_space.shape, ...)
        """
        self.obs[self._start_ind[epoch]:self._end_ind[epoch]] = value

    def view_ob(self, epoch):
        """View observation of an epoch."""
        return self.obs[self._start_ind[epoch]:self._end_ind[epoch]]

    def add_ob(self, epoch, value):
        """Add value to observation.

        Args:
            epoch: string, must be name of an added epoch
            value: np array (ob_space.shape, ...)
        """
        ob = self.view_ob(epoch)
        ob += value  # in-place

    def set_groundtruth(self, epoch, value):
        """Set groundtruth value."""
        self.gt[self._start_ind[epoch]: self._end_ind[epoch]] = value

    def in_epoch(self, epoch, t=None):
        """Check if current time or time t is in epoch"""
        if t is None:
            t = self.t  # Default
        return self._start_t[epoch] <= t < self._end_t[epoch]

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
        # TODO: Manually updating task.t here is bad shouldn't allow other things to change it
        obs, reward, done, info = self.task._step(action)
        self.task.t += self.task.dt  # increment within trial time count
        self.task.t_ind += 1

        if self.t >= self.tmax - self.dt:
            info['new_trial'] = True

        if info['new_trial']:
            self.task.t = self.task.t_ind = 0  # Reset within trial time count
            self.task.num_tr += 1  # Increment trial count
            self.new_trial(info=info)  # new_trial from wrapper
        return obs, reward, done, info

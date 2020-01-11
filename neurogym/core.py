#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import gym


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


class Env(BaseEnv):
    """The main Neurogym class for trial-based tasks."""

    def __init__(self, dt=100, num_trials_before_reset=10000000):
        super().__init__()
        self.dt = dt
        self.t = self.t_ind = 0
        self.tmax = 10000  # maximum time steps
        self.num_tr = 0
        self.num_tr_exp = num_trials_before_reset
        self.seed()

    def _step(self, action):
        """Private interface for the environment.

        Receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        raise NotImplementedError('_step is not defined by user.')

    def _new_trial(self):
        """Private interface for starting a new trial.

        Returns:
            trial_info: a dictionary of trial information
        """
        raise NotImplementedError('_new_trial is not defined by user.')

    def step(self, action):
        """Public interface for the environment."""
        obs, reward, done, info = self._step(action)

        self.t += self.dt  # increment within trial time count
        self.t_ind += 1

        if self.t >= self.tmax - self.dt:
            info['new_trial'] = True

        # TODO: Handle the case when new_trial is not provided in info
        if info['new_trial']:
            self.new_trial()
        return obs, reward, done, info

    def new_trial(self):
        """Public interface for starting a new trial."""
        self.t = self.t_ind = 0  # Reset within trial time count
        self.num_tr += 1  # Increment trial count
        return self._new_trial()  # Run user defined _new_trial method

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        self.num_tr = 0
        self.t = self.t_ind = 0

        # TODO: Check this works with wrapper
        self.trial = self.new_trial()
        obs, _, _, _ = self.step(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass


class EpochEnv(Env):
    """Environment class with trial/epoch structure."""

    def add_epoch(self, epoch, duration, start=None, before=None, after=None,
                  last_epoch=False
                  ):
        """Add an epoch.

        Args:
            epoch: string, name of the epoch
            duration: float, duration of the epoch
            start: start time of the epoch, float
            before: (optional) string, name of epoch that this epoch is before
            after: (optional) string, name of epoch that this epoch is after
            last_epoch: bool, default False. If True, then this is last epoch
                will generate self.tmax, self.tind, and self.obs
        """
        if after is not None:
            start = getattr(self, after + '_1')
        elif before is not None:
            start = getattr(self, before + '_0') - duration
        else:
            if start is None:
                raise ValueError('''start must be provided if
                before and after are None''')

        setattr(self, epoch + '_0', start)
        setattr(self, epoch + '_1', start + duration)
        setattr(self, epoch + '_ind0', int(start/self.dt))
        setattr(self, epoch + '_ind1', int((start + duration)/self.dt))

        if last_epoch:
            self._init_trial(start + duration)

    def _init_trial(self, tmax):
        """Initialize trial info with tmax, tind, obs"""
        tmax_ind = int(tmax/self.dt)
        self.tmax = tmax_ind * self.dt
        ob_shape = [tmax_ind] + list(self.observation_space.shape)
        self.obs = np.zeros(ob_shape)

    def set_ob(self, epoch, value):
        """Set observation in epoch to value.

        Args:
            epoch: string, must be name of an added epoch
            value: np array (ob_space.shape, ...)
        """
        self.obs[getattr(self, epoch+'_ind0'):getattr(self, epoch+'_ind1')] = value

    def add_ob(self, epoch, value):
        """Add value to observation.

        Args:
            epoch: string, must be name of an added epoch
            value: np array (ob_space.shape, ...)
        """
        self.obs[getattr(self, epoch+'_ind0'): getattr(self, epoch+'_ind1')] += value

    def in_epoch(self, epoch, t=None):
        """Check if time t is in epoch"""
        if t is None:
            t = self.t  # For backward compatibility
        return getattr(self, epoch+'_0') <= t < getattr(self, epoch+'_1')

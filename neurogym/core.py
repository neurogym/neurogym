#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
import gym


class Env(gym.Env):
    """The main Neurogym class specifying the basic structure of all tasks"""

    def __init__(self, dt=100):
        super().__init__()
        self.dt = dt
        self.t = 0
        self.num_tr = 0
        self.perf = 0
        self.num_tr_perf = 0
        # TODO: make this a parameter
        self.num_tr_exp = 10000000  # num trials after which done = True
        self.seed()

    def step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information (info). Aditionally, if the current trial is done
        (info['new_trial']==True) it calls the function _new_trial.
        """
        return None, None, None, None

    def reset(self):
        """
        restarts the experiment with the same parameters
        """
        self.perf = 0
        self.num_tr_perf = 0
        self.num_tr = 0
        self.t = 0

        self.trial = self._new_trial()
        obs, _, _, _ = self.step(self.action_space.sample())
        return obs

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        pass

    # Auxiliary functions
    def seed(self, seed=None):
        self.rng = random
        self.rng.seed(seed)
        return [seed]

    def _step(self, action):
        """
        receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information
        """
        return None, None, None, None

    def _new_trial(self):
        """Starts a new trial within the current experiment.

        Returns:
            trial_info: a dictionary of trial information
        """
        return {}

    def in_epoch(self, t, epoch):
        """Check if t is in epoch."""
        dur = self.trial['durations']
        return (dur[epoch][0] <= t < dur[epoch][1])



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

    def in_epoch(self, epoch, t):
        """Check if time t is in epoch"""
        return getattr(self, epoch+'_0') <= t < getattr(self, epoch+'_1')

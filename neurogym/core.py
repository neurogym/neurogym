#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import gym
import random


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

    def add_epoch(self, name, start, duration):
        """Add an epoch.

        Args:
            name: string, name of the epoch
            start: start time of the epoch, float or string
                if string, then start from the end of another epoch
            duration: float, duration of the epoch
        """
        setattr(self, name + '_0', start)
        setattr(self, name + '_1', start + duration)


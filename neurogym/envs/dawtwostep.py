#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daw two-step task

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


class DawTwoStep(ngym.TrialEnv):
    def __init__(self, dt=100, timing=()):
        super().__init__(dt=dt)
        # Actions ('FIXATE', 'ACTION1', 'ACTION2')
        self.actions = [0, 1, 2]

        # trial conditions
        self.p1 = 0.8  # probability of transitioning to state1 with action1
        self.p2 = 0.8  # probability of transitioning to state2 with action2
        self.p_switch = 0.025  # switch reward contingency
        self.high_reward_p = 0.9
        self.low_reward_p = 0.1
        self.tmax = 3*self.dt
        self.mean_trial_duration = self.tmax
        self.state1_high_reward = self.rng.random() > 0.5

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def __str__(self):
        string = 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration / self.dt)
        return string

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        # determine the transitions
        transition = np.empty((3,))
        st1 = 1
        st2 = 2
        tmp1 = st1 if self.rng.random() < self.p1 else st2
        tmp2 = st2 if self.rng.random() < self.p2 else st1
        transition[self.actions[1]] = tmp1
        transition[self.actions[2]] = tmp2

        # swtich reward contingency
        switch = self.rng.random() < self.p_switch
        if switch:
            self.state1_high_reward = not self.state1_high_reward
        # which state to reward with more probability
        if self.state1_high_reward:
            hi_state, low_state = 0, 1
        else:
            hi_state, low_state = 1, 0

        reward = np.empty((2,))
        reward[hi_state] = (self.rng.random() <
                            self.high_reward_p) * self.R_CORRECT
        reward[low_state] = (self.rng.random() <
                             self.low_reward_p) * self.R_CORRECT

        return {
            'transition':  transition,
            'reward': reward
            }

    def _step(self, action):
        trial = self.trial
        info = {'new_trial': False, 'gt': np.zeros((3,))}
        reward = 0

        obs = np.zeros((3,))
        if self.t == 0:  # at stage 1, if action==fixate, abort
            if action == 0:
                reward = self.R_ABORTED
                info['new_trial'] = True
            else:
                state = trial['transition'][action]
                obs[int(state)] = 1
                reward = trial['reward'][int(state-1)]
        elif self.t == self.dt:
            obs[0] = 1
            if action != 0:
                reward = self.R_ABORTED
            info['new_trial'] = True
        else:
            raise ValueError('t is not 0 or 1')

        return obs, reward, False, info

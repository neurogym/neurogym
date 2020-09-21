#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


# TODO: Need better description
class DawTwoStep(ngym.TrialEnv):
    """Daw Two-step task.

    On each trial, an initial choice between two options lead
    to either of two, second-stage states. In turn, these both
    demand another two-option choice, each of which is associated
    with a different chance of receiving reward.
    """
    metadata = {
        'paper_link': 'https://www.sciencedirect.com/science/article/' +
        'pii/S0896627311001255',
        'paper_name': 'Model-Based Influences on Humans' +
        ' Choices and Striatal Prediction Errors',
        'tags': ['two-alternative']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)
        if timing is not None:
            print('Warning: Two-step task does not require timing variable.')
        # Actions ('FIXATE', 'ACTION1', 'ACTION2')
        self.actions = [0, 1, 2]

        # trial conditions
        self.p1 = 0.8  # prob of transitioning to state1 with action1 (>=05)
        self.p2 = 0.8  # prob of transitioning to state2 with action2 (>=05)
        self.p_switch = 0.025  # switch reward contingency
        self.high_reward_p = 0.9
        self.low_reward_p = 0.1
        self.tmax = 3*self.dt
        self.mean_trial_duration = self.tmax
        self.state1_high_reward = True
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # determine the transitions
        transition = np.empty((3,))
        st1 = 1
        st2 = 2
        tmp1 = st1 if self.rng.rand() < self.p1 else st2
        tmp2 = st2 if self.rng.rand() < self.p2 else st1
        transition[self.actions[1]] = tmp1
        transition[self.actions[2]] = tmp2

        # swtich reward contingency
        switch = self.rng.rand() < self.p_switch
        if switch:
            self.state1_high_reward = not self.state1_high_reward
        # which state to reward with more probability
        if self.state1_high_reward:
            hi_state, low_state = 0, 1
        else:
            hi_state, low_state = 1, 0

        reward = np.empty((2,))
        reward[hi_state] = (self.rng.rand() <
                            self.high_reward_p) * self.rewards['correct']
        reward[low_state] = (self.rng.rand() <
                             self.low_reward_p) * self.rewards['correct']
        self.ground_truth = hi_state+1  # assuming p1, p2 >= 0.5
        trial = {
            'transition':  transition,
            'reward': reward,
            'hi_state': hi_state,
            }

        return trial

    def _step(self, action):
        trial = self.trial
        info = {'new_trial': False}
        reward = 0

        ob = np.zeros((3,))
        if self.t == 0:  # at stage 1, if action==fixate, abort
            if action == 0:
                reward = self.rewards['abort']
                info['new_trial'] = True
            else:
                state = trial['transition'][action]
                ob[int(state)] = 1
                reward = trial['reward'][int(state-1)]
                self.performance = action == self.ground_truth
        elif self.t == self.dt:
            ob[0] = 1
            if action != 0:
                reward = self.rewards['abort']
            info['new_trial'] = True
        else:
            raise ValueError('t is not 0 or 1')

        return ob, reward, False, info


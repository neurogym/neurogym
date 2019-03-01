#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daw two-step task

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class DawTwoStep(ngym.ngym):
    # TODO: remain to be carefully tested
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'STATE1', 'STATE2')

    # Actions
    actions = tasktools.to_map('FIXATE', 'ACTION1', 'ACTION2')

    # Trial conditions
    p1 = 0.8  # first stage transition probability
    p2 = 0.8  # second stage transition probability
    # TODO: this should be implemented through the wrapper
    p_switch = 0.025  # switch reward contingency
    high_reward_p = 0.9
    low_reward_p = 0.1
    state1_high_reward = True
    tmax = 2

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.trial = self._new_trial()
        print('------------------------')
        print('Daw two-step task')
        print('time step (ignored): ' + str(self.dt))
        print('------------------------')

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        # determine the transitions
        transition = dict()
        tmp1 = self.inputs['STATE1'] if self.rng.rand() < self.p1 else self.inputs['STATE2']
        tmp2 = self.inputs['STATE2'] if self.rng.rand() < self.p1 else self.inputs['STATE1']
        transition[self.actions['ACTION1']] = tmp1
        transition[self.actions['ACTION2']] = tmp2

        if self.state1_high_reward:
            hi_state, low_state = 'STATE1', 'STATE2'
        else:
            hi_state, low_state = 'STATE2', 'STATE1'

        reward = dict()
        reward[self.inputs[hi_state]] = (self.rng.rand() < self.high_reward_p) * self.R_CORRECT
        reward[self.inputs[low_state]] = (self.rng.rand() < self.low_reward_p) * self.R_CORRECT

        return {
            'transition':  transition,
            'reward': reward
            }

    def step(self, action):
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False

        # TODO: should we section by reward/input or epochs?
        obs = np.zeros(len(self.inputs))
        if self.t == 0:  # at stage 1
            if action == self.actions['FIXATE']:
                reward = self.R_ABORTED
                info['continue'] = False
            else:
                state = trial['transition'][action]
                obs[state] = 1
                reward = trial['reward'][state]
        elif self.t == 1:
            obs[self.inputs['FIXATION']] = 1
            if action != self.actions['FIXATE']:
                reward = self.R_ABORTED
            info['continue'] = False
        else:
            raise ValueError('t is not 0 or 1')

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += 1

        done = False  # TODO: revisit
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matching Penny task
See Daeyeol Lee's papers
TODO: add the actual papers
"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class MatchingPenny(ngym.ngym):
    def __init__(self, dt=100, opponent_type=None):
        super().__init__(dt=dt)
        # TODO: remain to be carefully tested
        # Opponent Type
        self.opponent_type = opponent_type

        # Rewards
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # TODO: Add more types of opponents
        # determine the transitions
        if self.opponent_type is None:
            opponent_action = int(self.rng.rand() > 0.5)
        else:
            raise NotImplementedError('Opponent type {:s} not implemented'.
                                      format(self.opponent_type))

        return {
            'opponent_action': opponent_action,
            }

    def _step(self, action):
        trial = self.trial
        tr_perf = True
        obs = np.zeros(self.observation_space.shape)
        obs[trial['opponent_action']] = 1.
        if action == trial['opponent_action']:
            reward = self.R_CORRECT
        else:
            reward = self.R_FAIL

        # ---------------------------------------------------------------------
        # new trial?
        new_trial = True
        info = {'new_trial': True}
        self.t = 0
        self.num_tr += 1
        # compute perf
        self.perf, self.num_tr_perf =\
            tasktools.compute_perf(self.perf, reward,
                                   self.num_tr_perf, tr_perf)

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

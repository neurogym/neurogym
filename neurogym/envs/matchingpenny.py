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
import neurogym as ngym


class MatchingPenny(ngym.TrialEnv):
    metadata = {
        'description': 'The agent is rewarded when it selects the same target as the computer,',
        'paper_link': 'https://www.nature.com/articles/nn1209',
        'paper_name': '''Prefrontal cortex and decision making in a mixed-strategy game''',
    }

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

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial (trials are one step long)
        # ---------------------------------------------------------------------
        # TODO: Add more types of opponents
        # determine the transitions
        if self.opponent_type is None:
            opponent_action = int(self.rng.random() > 0.5)
        else:
            raise ValueError('Unknown opponent type {:s}'.format(self.opponent_type))

        self.trial = {
            'opponent_action': opponent_action,
            }

    def _step(self, action):
        trial = self.trial
        obs = np.zeros(self.observation_space.shape)
        obs[trial['opponent_action']] = 1.
        if action == trial['opponent_action']:
            reward = self.R_CORRECT
        else:
            reward = self.R_FAIL

        info = {'new_trial': True, 'gt': obs}
        return obs, reward, False, info

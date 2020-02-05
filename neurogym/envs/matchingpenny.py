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
        'description': '''The agent is rewarded when it selects the
         same target as the computer.''',
        'paper_link': 'https://www.nature.com/articles/nn1209',
        'paper_name': '''Prefrontal cortex and decision making in a
         mixed-strategy game''',
        'tags': ['two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, opponent_type='random',
                 timing=None):
        """
        The agent is rewarded when it selects the same target as the computer.
        dt: Timestep duration. (def: 100 (ms), int)
        opponent_type: Type of opponent. (def: 'random', str)
        rewards:
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: 0., float)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt)
        if timing is not None:
            print('Warning: Matching-Penny task does not require' +
                  ' timing variable.')
        # TODO: remain to be carefully tested
        # Opponent Type
        self.opponent_type = opponent_type

        # Rewards
        reward_default = {'R_CORRECT': +1., 'R_FAIL': 0.}
        if rewards is not None:
            reward_default.update(rewards)
        self.R_CORRECT = reward_default['R_CORRECT']
        self.R_FAIL = reward_default['R_FAIL']

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial (trials are one step long)
        # ---------------------------------------------------------------------
        # TODO: Add more types of opponents
        # determine the transitions
        if self.opponent_type == 'random':
            opponent_action = int(self.rng.random() > 0.5)
        else:
            ot = self.opponent_type
            raise ValueError('Unknown opponent type {:s}'.format(ot))

        self.trial = {
            'opponent_action': opponent_action,
            }
        self.obs = np.zeros((1, self.observation_space.shape[0]))
        self.obs[0, opponent_action] = 1.
        self.gt = np.array([opponent_action])

    def _step(self, action):
        trial = self.trial
        obs = self.obs[0]
        if action == trial['opponent_action']:
            reward = self.R_CORRECT
        else:
            reward = self.R_FAIL

        info = {'new_trial': True, 'gt': self.gt}
        return obs, reward, False, info

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:41:50 2019

@author: molano
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:48:59 2019

@author: molano
"""

from gym.core import Wrapper
from neurogym.ops import tasktools


class SideBias(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, prob=(.2, .8), block_dur=200):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.prob = prob
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.env.trial['ground_truth']

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        trial = self.env._new_trial()
        # change rep. prob. every self.block_dur trials
        if self.env.num_tr % self.block_dur == 0:
            self.curr_block = int(not self.curr_block)

        probs = (1-self.prob[self.curr_block],
                 self.prob[self.curr_block])

        trial['ground_truth'] = self.env.rng.choices(self.env.choices,
                                                     weights=probs)[0]
        return trial

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)

        if info['new_trial']:
            info['prob'] = self.prob[self.curr_block]
            self.env.trial = self._new_trial()

        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = tasktools.choice(self.env.rng, [0, 1])

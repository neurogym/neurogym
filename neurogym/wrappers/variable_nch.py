#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import neurogym as ngym
import numpy as np
from neurogym.core import TrialWrapper
import warnings


class Variable_nch(TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_nch trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, block_nch=100, blocks_probs=None, sorted_ch=True,
                 prob_12=None):
        """
        block_nch: duration of each block containing a specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)

        assert isinstance(block_nch, int), 'block_nch must be integer'
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        self.block_nch = block_nch
        self.max_nch = len(self.unwrapped.choices)  # Max number of choices
        self.prob_12 = prob_12 if self.max_nch > 2 else 1
        self.sorted_ch = sorted_ch
        # uniform distr. across choices unless prob(n_ch=2) (prob_2) is specified
        if blocks_probs is not None:
            self.prob = blocks_probs[:self.max_nch-1]
            if np.sum(self.prob) == 0:
                self.prob = [1/(self.max_nch-1)]*(self.max_nch-1)
            else:
                self.prob = self.prob/np.sum(self.prob)
        else:
            self.prob = [1/(self.max_nch-1)]*(self.max_nch-1)
        # Initialize with a random number of active choices (never 1)
        self.nch = self.rng.choice(range(2, self.max_nch + 1), p=self.prob)

    def new_trial(self, **kwargs):

        if 'ground_truth' in kwargs.keys():
            warnings.warn('Variable_nch wrapper ' +
                          'will ignore passed ground truth')
        # We change number of active choices every 'block_nch'.
        if self.unwrapped.num_tr % self.block_nch == 0:
            fx_12 = self.prob_12 is not None
            if fx_12 and self.unwrapped.rng.rand() < self.prob_12:
                self.nch = 2
                self.sel_chs = np.arange(self.nch)
            else:
                if self.sorted_ch:
                    prb = self.prob[1*fx_12:]
                    self.nch = self.rng.choice(range(2+1*fx_12, self.max_nch + 1),
                                               p=prb/np.sum(prb))
                    self.sel_chs = np.arange(self.nch)
                else:
                    self.nch = self.rng.choice(range(2, self.max_nch + 1),
                                               p=self.prob)
                    self.sel_chs = sorted(self.rng.choice(range(self.max_nch),
                                                          self.nch, replace=False))
                    while (fx_12 and set(self.sel_chs) == set(np.arange(2))):
                        self.sel_chs = sorted(self.rng.choice(range(self.max_nch),
                                                              self.nch,
                                                              replace=False))
        kwargs.update({'sel_chs': self.sel_chs})
        self.env.new_trial(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['nch'] = self.nch
        info['sel_chs'] = '-'.join([str(x+1) for x in self.sel_chs])
        return obs, reward, done, info

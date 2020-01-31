#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:48:59 2019

@author: molano
"""

import neurogym as ngym


class TrialHistory(ngym.TrialWrapper):
    metadata = {
        'description': 'Change ground truth probability ' +
        'based on previous outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': '''Response outcomes gate the impact of expectations
         on perceptual decisions''',
        'rep_prob': '''Specifies probabilities of repeating for each block.
        (def: (.2, .8))''',
        'block_dur': 'Number of trials per block. (def: 200 (int))',
        'blk_ch_prob': '''If not None, specifies the probability of changing
        block (randomly). (def: None)''',
    }

    def __init__(self, env, rep_prob=(.2, .8), block_dur=200,
                 blk_ch_prob=None):
        super().__init__(env)

        self.rep_prob = rep_prob
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = -1
        self.blk_ch_prob = blk_ch_prob

        if len(self.task.choices) != 2:
            raise ValueError('Currently, this wrapper only works this 2AFC tasks.')

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % len(self.rep_prob)

        rep_prob = self.rep_prob[self.curr_block]

        if self.prev_trial == self.task.choices[0]:
            probs = (rep_prob, 1-rep_prob)
        else:
            probs = (1-rep_prob, rep_prob)

        ground_truth = self.task.rng.choices(self.task.choices,
                                             weights=probs)[0]
        self.prev_trial = ground_truth
        kwargs.update({'ground_truth': ground_truth})
        self.env.new_trial(**kwargs)

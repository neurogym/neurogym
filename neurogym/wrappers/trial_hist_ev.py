#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:02:28 2020

@author: molano
"""

import neurogym as ngym
from neurogym.core import TrialWrapperV2
import numpy as np


class TrialHistoryEvolution(TrialWrapperV2):
    """
    This wrapper imposes specific probability transition matrices that are
    characterized by probs (the probability of the most likely next choice).
    The number of transition matrices is specified by num_contexts. The
    transition matrices are reset with a probability specified by death_prob.

    Parameters
    ----------
    env : neurogym.env
        Environment that will be wrapped
    probs : float, optional
        The probability of the the most likely next choice. The default is None.
    ctx_dur : int, optional
        Duration of the contexts (if ctx_ch_prob is None). Default is 200 (trials).
    num_contexts : int, optional
        Number of contexts experienced by each individual. The default is 3.
    death_prob : float, optional
        Probability of death by each individual. The default is 0.0001.
    ctx_ch_prob : float, optional
        Probability of context change. The default is None.
    balanced_probs : boolean, optional
        Indicates whether transtion matrices are balanced. The default is False.

    Raises
    ------
    AttributeError
        DESCRIPTION.

    Returns
    -------
    wrapped environment

    """
    metadata = {
        'description': 'Change ground truth probability based on previous' +
        'outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': 'Response outcomes gate the impact of expectations ' +
        'on perceptual decisions'
    }

    def __init__(self, env, probs=None, ctx_dur=200, num_contexts=3,
                 death_prob=0.0001, ctx_ch_prob=None, balanced_probs=False):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
            self.th_choices = self.unwrapped.choices
            self.curr_n_ch = self.n_ch
        except AttributeError:
            raise AttributeError('''SideBias requires task
                                 to have attribute choices''')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        assert probs is not None, 'Please provide choices probabilities'
        self.probs = probs
        self.balanced_probs = balanced_probs
        self.num_contexts = num_contexts
        self.death_prob = death_prob
        self.curr_contexts = self.contexts
        self.curr_tr_mat = self.trans_probs
        assert self.curr_tr_mat.shape[1] == self.n_ch,\
            'The number of choices {:d}'.format(self.tr_mat.shape[1]) +\
            ' inferred from prob mismatchs {:d}'.format(self.n_ch) +\
            ' inferred from choices'
        self.ctx_dur = ctx_dur
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization
        self.ctx_ch_prob = ctx_ch_prob

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        block_already_changed = False
        # Check if n_ch is passed and if it is different from previous value
        if 'n_ch' in kwargs.keys() and kwargs['n_ch'] != self.curr_n_ch:
            self.curr_n_ch = kwargs['n_ch']
            self.prev_trial = self.rng.choice(self.th_choices[:self.curr_n_ch])
            self.curr_tr_mat = self.trans_probs
            block_already_changed = True

        # change rep. prob. every self.ctx_dur trials
        if not block_already_changed:
            if self.ctx_ch_prob is None:
                block_change = self.unwrapped.num_tr % self.ctx_dur == 0
            else:
                block_change = self.unwrapped.rng.rand() < self.ctx_ch_prob
            if block_change:
                self.curr_tr_mat = self.trans_probs

        probs_curr_blk = self.curr_tr_mat[self.curr_block, self.prev_trial, :]
        ground_truth = self.unwrapped.rng.choice(self.th_choices[:self.curr_n_ch],
                                                 p=probs_curr_blk)
        self.prev_trial =\
            np.where(self.th_choices[:self.curr_n_ch] == ground_truth)[0][0]
        kwargs.update({'ground_truth': ground_truth,
                       'curr_block': self.curr_block})
        self.env.new_trial(**kwargs)

    @property
    def trans_probs(self):
        '''
        if prob is float it creates the transition matrix
        if prob is already a matrix it normalizes the probabilities and extracts
        the subset corresponding to the current number of choices
        '''
        if self.unwrapped.rng.rand() < self.death_prob/self.ctx_ch_prob:
            self.curr_contexts = self.contexts
        context = self.curr_contexts[self.rng.choice(range(self.num_contexts))]
        tr_mat = np.eye(self.curr_n_ch)*self.probs
        tr_mat[tr_mat == 0] = (1-self.probs)/(self.curr_n_ch-1)
        tr_mat = tr_mat[context, :]
        tr_mat = np.expand_dims(tr_mat, axis=0)
        tr_mat = np.unique(tr_mat, axis=0)
        self.curr_n_blocks = tr_mat.shape[0]
        self.curr_block = self.unwrapped.rng.choice(range(self.curr_n_blocks))
        self.blk_id = int(''.join([str(x+1) for x in context]))
        return tr_mat

    @property
    def contexts(self):
        contexts = np.empty((self.num_contexts, self.n_ch))
        for i_ctx in range(self.num_contexts):
            if self.balanced_probs:
                indx = np.arange(self.curr_n_ch)
                np.random.shuffle(indx)
            else:
                indx = np.random.choice(self.curr_n_ch, size=(self.curr_n_ch,))
            contexts[i_ctx, :] = indx
        return contexts.astype(int)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['curr_block'] = self.blk_id
        return obs, reward, done, info

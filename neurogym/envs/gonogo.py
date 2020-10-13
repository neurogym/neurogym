#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class GoNogo(ngym.TrialEnv):
    r"""Go/No-go task.

    A stimulus is shown during the stimulus period. The stimulus period is
    followed by a delay period, and then a decision period. If the stimulus is
    a Go stimulus, then the subject should choose the action Go during the
    decision period, otherwise, the subject should remain fixation.
    """
    # TODO: Find the original go-no-go paper
    metadata = {
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'tags': ['delayed response', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)
        # Actions (fixate, go)
        self.actions = [0, 1]
        # trial conditions
        self.choices = [0, 1]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.5, 'miss': -0.5}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 0,
            'stimulus': 500,
            'delay': 500,
            'decision': 500}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation spaces
        name = {'fixation': 0, 'nogo': 1, 'go': 2}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32, name=name)
        self.action_space = spaces.Discrete(2, {'fixation': 0, 'go': 1})

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'ground_truth': self.rng.choice(self.choices)
        }
        trial.update(kwargs)

        # Period info
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods)
        # set observations
        self.add_ob(1, where='fixation')
        self.add_ob(1, 'stimulus', where=trial['ground_truth']+1)
        self.set_ob(0, 'decision')
        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards['miss']*trial['ground_truth']
        self.performance = 1-trial['ground_truth']
        # set ground truth during decision period
        self.set_groundtruth(trial['ground_truth'], 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if gt != 0:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
                    self.performance = 0

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}

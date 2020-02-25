#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:01:22 2019

@author: molano

GO/NO-GO task based on:

  Active information maintenance in working memory by a sensory cortex
  Xiaoxing Zhang, Wenjun Yan, Wenliang Wang, Hongmei Fan, Ruiqing Hou,
  Yulei Chen, Zhaoqin Chen, Shumin Duan, Albert Compte, Chengyu Li bioRxiv 2018

  https://www.biorxiv.org/content/10.1101/385393v1

"""
from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym


class GoNogo(ngym.PeriodEnv):
    # TODO: Find the original go-no-go paper
    metadata = {
        'description': 'Go/No-Go task in which the subject has either Go' +
        ' (e.g. lick) or not Go depending on which one of two stimuli is' +
        ' presented with.',
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'timing': {
            'fixation': ('constant', 0),
            'stimulus': ('constant', 500),
            'resp_delay': ('constant', 500),
            'decision': ('constant', 500)},
        'tags': ['delayed response', 'go-no-go', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        Go/No-Go task in which the subject has either Go (e.g. lick)
        or not Go depending on which one of two stimuli is presented with.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards: reward dictionary
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt, timing=timing)
        # Actions (fixate, go)
        self.actions = [0, 1]
        # trial conditions
        self.choices = [0, 1]

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -0.5, 'miss': -0.5}
        if rewards:
            self.rewards.update(rewards)

        self.abort = False
        # set action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.act_dict = {'fixation': 0, 'go': 1}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'nogo': 1, 'go': 2}

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices)
        }
        self.trial.update(kwargs)

        # Period info
        periods = ['fixation', 'stimulus', 'resp_delay', 'decision']
        self.add_period(periods, after=0, last_period=True)
        # set observations
        self.add_ob(1, where='fixation')
        self.add_ob(1, 'stimulus', where=self.trial['ground_truth']+1)
        self.set_ob(0, 'decision')
        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards['miss']*self.trial['ground_truth']
        self.performance = 1-self.trial['ground_truth']
        # set ground truth during decision period
        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def _step(self, action):
        new_trial = False
        reward = 0
        obs = self.obs_now
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

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = GoNogo()
    ngym.utils.plot_env(env, def_act=0)
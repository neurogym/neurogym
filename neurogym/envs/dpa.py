#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:25:08 2019

@author: molano


Delay Pair Association (DPA) task based on:

  Active information maintenance in working memory by a sensory cortex
  Xiaoxing Zhang, Wenjun Yan, Wenliang Wang, Hongmei Fan, Ruiqing Hou,
  Yulei Chen, Zhaoqin Chen, Shumin Duan, Albert Compte, Chengyu Li bioRxiv 2018

  https://www.biorxiv.org/content/10.1101/385393v1

"""
import numpy as np
from neurogym.ops import tasktools
import neurogym as ngym
from gym import spaces


# TODO: This task is obsolete
class DPA(ngym.EpochEnv):
    def __init__(self, dt=100,
                 timing=(100, 200, 600, 600, 200, 100, 100)):
        # call ngm __init__ function
        super().__init__(dt=dt)
        # Actions
        self.actions = [-1, 1]
        # trial conditions
        self.dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        # Input noise
        self.sigma = np.sqrt(2*100*0.001)
        # Epoch durations
        self.fixation = timing[0]
        self.dpa1 = timing[1]
        self.delay_min = timing[2]
        self.delay_max = timing[3]
        self.stimulus_mean = (timing[2]+timing[3])/2
        self.dpa2 = timing[4]
        self.resp_delay = timing[5]
        self.decision = timing[6]
        self.delay_mean = (self.delay_min + self.delay_max)/2
        self.mean_trial_duration = self.fixation + self.dpa1 +\
            self.delay_mean + self.dpa2 + self.resp_delay + self.decision

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_INCORRECT = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1., 1, shape=(5, ),
                                            dtype=np.float32)

    def __str__(self):
        string = ''
        if self.fixation == 0 or self.decision == 0 or self.stimulus_mean == 0:
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of the fixation, stimulus and decision '
            string += 'periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration/self.dt)
        return string

    def _new_trial(self):
        pair = self.rng.choice(self.dpa_pairs)

        if np.diff(pair)[0] == 2:
            ground_truth = 0
        else:
            ground_truth = 1

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)

        self.add_epoch('fixation', duration=self.fixation, after=0)
        self.add_epoch('dpa1', duration=self.dpa1, after='fixation')
        self.add_epoch('delay', duration=delay, after='dpa1')
        self.add_epoch('dpa2', duration=self.dpa1, after='delay')
        self.add_epoch('resp_delay', duration=self.resp_delay, after='dpa2')
        self.add_epoch('decision', duration=self.decision, after='resp_delay', last_epoch=True)

        dpa1, dpa2 = pair
        tmp = [0] * 5
        tmp[dpa1] = 1
        self.set_ob('dpa1', tmp)
        tmp = [0] * 5
        tmp[dpa2] = 1
        self.set_ob('dpa2', tmp)

        self.set_groundtruth('decision', ground_truth)

        return {
            'ground_truth':     ground_truth,
            'pair':     pair
            }

    def _step(self, action):
        info = {'new_trial': False}
        reward = 0
        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]
        if self.in_epoch('fixation'):
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action == 1:
                info['new_trial'] = True
                if gt == 1:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT

        return obs, reward, False, info

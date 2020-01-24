#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:41:52 2019

@author: molano
"""

from gym.core import Wrapper
import os
import numpy as np


class ManageData(Wrapper):
    metadata = {
        'description': '''saves relevant behavioral information: rewards,
         actions, observations, new trial, ground truth''',
        'paper_link': None,
        'paper_name': None,
        'folder': None,
    }

    def __init__(self, env, folder=None, num_tr_save=100000):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.num_tr = 0
        # data to save
        self.choice_mat = []
        self.gt_mat = []
        self.stim_mat = []
        self.reward_mat = []
        self.cum_obs = 0
        self.cum_rew = 0
        if folder is not None:
            self.folder = folder + '/'
        else:
            self.folder = "/tmp/"
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        # seeding
        self.saving_name = self.folder +\
            self.env.__class__.__name__ + str(self.inst)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.cum_obs += obs
        self.cum_rew += rew
        if info['new_trial']:
            self.num_tr += 1
            self.choice_mat.append(action)
            self.stim_mat.append(self.cum_obs)
            self.cum_obs = 0
            self.reward_mat.append(self.cum_rew)
            self.cum_rew = 0
            if 'gt' in info.keys():
                gt = np.argmax(info['gt'])
                self.gt_mat.append(gt)

            # save data
            if self.num_tr % self.num_tr_save == 0:
                data = {'choice': self.choice_mat,
                        'stimulus': self.stim_mat,
                        'correct_side': self.gt_mat,
                        'reward': self.reward_mat}

                np.savez(self.saving_name + '_bhvr_data_' +
                         str(self.num_tr) + '.npz', **data)
                if self.plt_tr:
                    self.render()

                self.choice_mat = []
                self.gt_mat = []
                self.stim_mat = []
                self.reward_mat = []
        return obs, rew, done, info

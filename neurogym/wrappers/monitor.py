#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:41:52 2019

@author: molano
"""

from gym.core import Wrapper
import os
import numpy as np


class Monitor(Wrapper):
    metadata = {
        'description': 'Saves relevant behavioral information: rewards,' +
        ' actions, observations, new trial, ground truth.',
        'paper_link': None,
        'paper_name': None,
        'folder': 'Folder where the data will be saved. (def: None)',
        'num_tr_save': '''Data will be saved every num_tr_save trials.
        (def: 100000)''',
        'verbose': 'Whether to print information about average reward and' +
        ' number of trials',
        'info_keywords': '(tuple) extra information to log, from the ' +
        'information return of environment.step'
    }

    def __init__(self, env, folder=None, num_tr_save=100000, verbose=False,
                 info_keywords=()):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.num_tr = 0
        # data to save
        self.info_keywords = info_keywords
        self.reset_data()
        self.cum_obs = 0
        self.cum_rew = 0
        self.num_tr_save = num_tr_save
        self.verbose = verbose
        if folder is not None:
            self.folder = folder + '/'
        else:
            self.folder = "/tmp/"
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        # seeding
        self.saving_name = self.folder +\
            self.env.__class__.__name__

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.cum_obs += obs
        self.cum_rew += rew
        if info['new_trial']:
            self.num_tr += 1
            self.data['choice'].append(action)
            self.data['stimulus'].append(self.cum_obs)
            self.cum_obs = 0
            self.data['reward'].append(self.cum_rew)
            self.cum_rew = 0
            if 'gt' in info.keys():
                gt = np.argmax(info['gt'])
                self.data['correct_side'].append(gt)
            for key in self.info_keywords:
                self.data[key].append(info[key])

            # save data
            if self.num_tr % self.num_tr_save == 0:
                np.savez(self.saving_name + '_bhvr_data_' +
                         str(self.num_tr) + '.npz', **self.data)
                if self.verbose:
                    print('--------------------')
                    print('Number of steps: ', np.mean(self.num_tr))
                    print('Average reward: ', np.mean(self.data['reward']))
                    for key in self.info_keywords:
                        print(key + ' : ' + str(info[key]))
                    print('--------------------')
                self.reset_data()
        return obs, rew, done, info

    def reset_data(self):
        data = {'choice': [], 'stimulus': [], 'correct_side': [], 'reward': []}
        for key in self.info_keywords:
            data[key] = []
        self.data = data

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:41:52 2019

@author: molano
"""

from gym.core import Wrapper
import os
import numpy as np
from neurogym.meta import info


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
        'sv_fig': 'Whether to save a figure of the experiment structure.' +
        ' If True, a figure will be updated every num_tr_save. (def: False)',
        'num_stps_sv_fig': 'Number of trial steps to include in the figure. ' +
        '(def: 100)'
    }
    # TODO: use names similar to Tensorboard

    def __init__(self, env, folder=None, num_tr_save=100000, verbose=False,
                 sv_fig=False, num_stps_sv_fig=100):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.num_tr = 0
        # data to save
        self.data = {'choice': [], 'stimulus': [], 'reward': []}
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
        # figure
        self.sv_fig = sv_fig
        if self.sv_fig:
            self.num_stps_sv_fig = num_stps_sv_fig
            self.stp_counter = 0
            self.obs_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.cum_obs += obs
        self.cum_rew += rew
        if self.sv_fig:
            self.store_data(obs, action, rew, info['gt'])
        if info['new_trial']:
            self.num_tr += 1
            self.data['choice'].append(action)
            self.data['stimulus'].append(self.cum_obs)
            self.cum_obs = 0
            self.data['reward'].append(self.cum_rew)
            self.cum_rew = 0
            for key in info:
                if key not in self.data.keys():
                    self.data[key] = [info[key]]
                else:
                    self.data[key].append(info[key])

            # save data
            if self.num_tr % self.num_tr_save == 0:
                np.savez(self.saving_name + '_bhvr_data_' +
                         str(self.num_tr) + '.npz', **self.data)
                if self.verbose:
                    print('--------------------')
                    print('Number of steps: ', np.mean(self.num_tr))
                    print('Average reward: ', np.mean(self.data['reward']))
                    print('--------------------')
                self.reset_data()
                if self.sv_fig:
                    self.stp_counter = 0
        return obs, rew, done, info

    def reset_data(self):
        for key in self.data.keys():
            self.data[key] = []

    def store_data(self, obs, action, rew, gt):
        if self.stp_counter <= self.num_stps_sv_fig:
            self.obs_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)
            self.gt_mat.append(gt)
            self.stp_counter += 1
        elif len(self.rew_mat) > 0:
            obs_mat = np.array(self.obs_mat)
            act_mat = np.array(self.act_mat)
            info.fig_(obs=obs_mat, actions=act_mat,
                      gt=self.gt_mat, rewards=self.rew_mat,
                      n_stps_plt=self.num_stps_sv_fig,
                      perf=self.data['reward'],
                      folder=self.folder)
            self.obs_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []

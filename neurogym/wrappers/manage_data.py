#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:41:52 2019

@author: molano
"""

from gym.core import Wrapper
import os
import numpy as np


class manage_data(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, inst=0, plt_tr=True, folder=None,
                 inst_to_save=[0], num_tr_save=100000):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.do = inst in inst_to_save
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if self.do:
            self.num_tr = 0
            self.inst = inst
            # data to save
            self.choice_mat = []
            self.gt_mat = []
            # for catch trials
            self.catch_tr_mat = []
            # for dual-task
            self.config_mat = []
            # for RDM + trial history
            self.rep_prob_mat = []
            self.tr_mat = []
            self.stim_mat = []
            self.reward_mat = []
            self.cum_obs = 0
            self.cum_rew = 0
            # for rendering
            self.obs_mat = []
            self.act_mat = []
            self.gt_mat_render = []
            self.rew_mat = []
            self.num_tr_save = num_tr_save
            self.max_num_samples = 200
            self.num_subplots = 3
            self.plt_tr = plt_tr and self.do
            if self.plt_tr:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                self.fig, self.ax = plt.subplots(self.num_subplots, 1)
            if folder is not None:
                self.folder = folder + '/'
            else:
                self.folder = "/tmp/"
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)
            # seeding
            self.env.seed()
            self.saving_name = self.folder +\
                self.env.__class__.__name__ + str(self.inst)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.do:
            self.cum_obs += obs
            self.cum_rew += rew
            self.store_data(obs, action, rew, info['gt'])
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
                if 'rep_prob' in info.keys():
                    self.rep_prob_mat.append(info['rep_prob'])
                if 'tr_mat' in info.keys():
                    self.tr_mat.append(info['tr_mat'])
                if 'config' in info.keys():
                    self.config_mat.append(info['config'])
                if 'catch_trial' in info.keys():
                    self.catch_tr_mat.append(info['catch_trial'])

                # save data
                if self.num_tr % self.num_tr_save == 0:
                    data = {'choice': self.choice_mat,
                            'stimulus': self.stim_mat,
                            'correct_side': self.gt_mat,
                            'reward': self.reward_mat}
                    if len(self.rep_prob_mat) != 0:
                        data['rep_prob'] = self.rep_prob_mat
                    if len(self.tr_mat) != 0:
                        data['tr_mat'] = self.tr_mat
                    if len(self.config_mat) != 0:
                        data['config'] = self.config_mat
                    if len(self.catch_tr_mat) != 0:
                        data['catch_trial'] = self.catch_tr_mat
                    np.savez(self.saving_name + '_bhvr_data_' +
                             str(self.num_tr) + '.npz', **data)
                    if self.plt_tr:
                        self.render()

                    self.choice_mat = []
                    self.gt_mat = []
                    self.config_mat = []
                    self.catch_tr_mat = []
                    self.rep_prob_mat = []
                    self.tr_mat = []
                    self.stim_mat = []
                    self.reward_mat = []
                    # for rendering
                    self.obs_mat = []
                    self.act_mat = []
                    self.gt_mat_render = []
                    self.rew_mat = []
        return obs, rew, done, info

    def render(self, mode='human'):
        """
        plots relevant variables/parameters
        """
        # observations
        obs = np.array(self.obs_mat)
        self.ax[0].imshow(obs.T, aspect='auto')
        self.ax[0].set_ylabel('obs')
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        # actions and ground truth
        self.ax[1].plot(np.arange(1, len(self.act_mat)+1)+0.5, self.act_mat)
        self.ax[1].set_ylabel('action (gt)')
        self.ax[1].set_xlim([0, self.max_num_samples+0.5])
        gt = np.array(self.gt_mat_render)
        self.ax[1].plot(np.arange(1, len(self.act_mat)+1)+0.5,
                        np.argmax(gt, axis=1), '--')
        self.ax[1].set_xlim([0, self.max_num_samples+0.5])
        self.ax[1].set_xticks([])
        self.ax[1].set_yticks([])
        # reward
        self.ax[2].plot(np.arange(1, len(self.act_mat)+1)+0.5, self.rew_mat)
        self.ax[2].set_ylabel('reward')
        self.ax[2].set_xlim([0, self.max_num_samples+0.5])
        self.fig.savefig(self.saving_name + '_trials.png')
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[2].cla()

    def store_data(self, obs, action, rew, gt):
        if len(self.rew_mat) < self.max_num_samples:
            self.obs_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)
            self.gt_mat_render.append(gt)

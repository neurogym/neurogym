#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym import Wrapper
import os
import numpy as np
from neurogym.utils.plotting import fig_


class Monitor(Wrapper):
    """Monitor task.

    Saves relevant behavioral information: rewards,actions, observations,
    new trial, ground truth.

    Args:
        folder: Folder where the data will be saved. (def: None, str)
            sv_per and sv_stp: Data will be saved every sv_per sv_stp's.
            (def: 100000, int)
        verbose: Whether to print information about average reward and number
            of trials. (def: False, bool)
        sv_fig: Whether to save a figure of the experiment structure. If True,
            a figure will be updated every sv_per. (def: False, bool)
        num_stps_sv_fig: Number of trial steps to include in the figure.
            (def: 100, int)
    """
    metadata = {
        'description': 'Saves relevant behavioral information: rewards,' +
        ' actions, observations, new trial, ground truth.',
        'paper_link': None,
        'paper_name': None,
    }
    # TODO: use names similar to Tensorboard

    def __init__(self, env, folder=None, sv_per=100000, sv_stp='trial',
                 verbose=False, sv_fig=False, num_stps_sv_fig=100, name='',
                 fig_type='png'):
        super().__init__(env)
        self.env = env
        self.num_tr = 0
        # data to save
        self.data = {'action': [], 'reward': []}
        self.sv_per = sv_per
        self.sv_stp = sv_stp
        self.fig_type = fig_type
        if self.sv_stp == 'timestep':
            self.t = 0
        self.verbose = verbose
        if folder is not None:
            self.folder = folder + '/'
        else:
            self.folder = "/tmp/"
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        # seeding
        self.sv_name = self.folder +\
            self.env.__class__.__name__+'_bhvr_data_'+name+'_'
        # figure
        self.sv_fig = sv_fig
        if self.sv_fig:
            self.num_stps_sv_fig = num_stps_sv_fig
            self.stp_counter = 0
            self.ob_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []
            self.perf_mat = []

    def reset(self, step_fn=None):
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(step_fn=step_fn)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if self.sv_fig:
            self.store_data(obs, action, rew, info)
        if self.sv_stp == 'timestep':
            self.t += 1
        if info['new_trial']:
            self.num_tr += 1
            self.data['action'].append(action)
            self.data['reward'].append(rew)
            for key in info:
                if key not in self.data.keys():
                    self.data[key] = [info[key]]
                else:
                    self.data[key].append(info[key])

            # save data
            save = False
            if self.sv_stp == 'timestep':
                save = self.t >= self.sv_per
            else:
                save = self.num_tr % self.sv_per == 0
            if save:
                np.savez(self.sv_name + str(self.num_tr) + '.npz', **self.data)
                if self.verbose:
                    print('--------------------')
                    print('Number of steps: ', np.mean(self.num_tr))
                    print('Average reward: ', np.mean(self.data['reward']))
                    print('--------------------')
                self.reset_data()
                if self.sv_fig:
                    self.stp_counter = 0
                if self.sv_stp == 'timestep':
                    self.t = 0
        return obs, rew, done, info

    def reset_data(self):
        for key in self.data.keys():
            self.data[key] = []

    def store_data(self, obs, action, rew, info):
        if self.stp_counter <= self.num_stps_sv_fig:
            self.ob_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)
            if 'gt' in info.keys():
                self.gt_mat.append(info['gt'])
            else:
                self.gt_mat.append(-1)
            if 'performance' in info.keys():
                self.perf_mat.append(info['performance'])
            else:
                self.perf_mat.append(-1)
            self.stp_counter += 1
        elif len(self.rew_mat) > 0:
            fname = self.sv_name + 'task_{0:06d}.'.format(self.num_tr)+self.fig_type
            obs_mat = np.array(self.ob_mat)
            act_mat = np.array(self.act_mat)
            fig_(ob=obs_mat, actions=act_mat,
                 gt=self.gt_mat, rewards=self.rew_mat,
                 performance=self.perf_mat,
                 fname=fname)
            self.ob_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []
            self.perf_mat = []

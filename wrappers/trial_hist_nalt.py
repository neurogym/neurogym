#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:15:02 2019

@author: molano
"""

from gym.core import Wrapper
import numpy as np
from neurogym.envs import nalt_rdm
import matplotlib.pyplot as plt


class TrialHistory_NAlt(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, n=2, tr_prob=0.8, block_dur=200,
                 blk_ch_prob=None, pass_blck=False):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        # buld transition matrix
        tr_mat = np.zeros((2, n, n)) + (1-tr_prob)/(n-1)
        for ind in range(n-1):
            tr_mat[0, ind, ind+1] = tr_prob
        tr_mat[0, n-1, 0] = tr_prob
        tr_mat[1, :, :] = tr_mat[0, :, :].T

        self.tr_mat = tr_mat
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])
        # duration of block (in number oif trials)
        self.block_dur = block_dur
        self.prev_trial = self.task.ground_truth
        self.blk_ch_prob = blk_ch_prob
        self.pass_blck_info = pass_blck

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % self.tr_mat.shape[0]
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % self.tr_mat.shape[0]
        # get probs
        probs = self.tr_mat[self.curr_block, self.prev_trial-1, :]
        ground_truth = self.task.rng.choices(self.task.choices,
                                             weights=probs)[0]
        self.prev_trial = ground_truth
        kwargs.update({'gt': ground_truth})
        self.env.new_trial(**kwargs)

    def reset(self):
        return self.task.reset()

    def _step(self, action):
        return self.env._step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            info['tr_mat'] = self.tr_mat[self.curr_block, :, :]
            self.prev_correct = reward == self.task.R_CORRECT
            self.new_trial()
        if self.pass_blck_info:
            obs = np.concatenate((obs, np.array([self.curr_block])))
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])


if __name__ == '__main__':
    env = nalt_rdm.nalt_RDM(timing=[100, 200, 200, 200, 100])
    env = TrialHistory_NAlt(env)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    num_steps_env = 1000
    for stp in range(int(num_steps_env)):
        action = 1  # env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        observations.append(obs)
        if info['new_trial']:
            actions_end_of_trial.append(action)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
        if 'config' in info.keys():
            config_mat.append(info['config'])
        else:
            config_mat.append([0, 0])

    rows = 3
    obs = np.array(observations)
    plt.figure()
    plt.imshow(tr_mat[0, :, :], aspect='auto')
    plt.figure()
    plt.subplot(rows, 1, 1)
    plt.imshow(obs.T, aspect='auto')
    plt.title('observations')
    plt.subplot(rows, 1, 2)
    plt.plot(actions, marker='+')
    #    plt.plot(actions_end_of_trial, '--')
    gt = np.array(gt)
    plt.plot(np.argmax(gt, axis=1), 'r')
    #    # aux = np.argmax(obs, axis=1)
    # aux[np.sum(obs, axis=1) == 0] = -1
    # plt.plot(aux, '--k')
    plt.title('actions')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.subplot(rows, 1, 3)
    plt.plot(rewards, 'r')
    plt.title('reward')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.show()

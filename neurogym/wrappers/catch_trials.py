#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:23:36 2019

@author: molano
"""
from gym.core import Wrapper
from neurogym.envs import nalt_rdm
from neurogym.wrappers import trial_hist_nalt
import numpy as np
import matplotlib.pyplot as plt


class CatchTrials(Wrapper):
    metadata = {
        'description': """Introduces catch trials in which the reward for
         a correct choice is modified (e.g. is set to the reward for an
         incorrect choice). Note that the wrapper only changes the reward
         associated to a correct answer and does not change the ground truth.
         Thus, the catch trial affect a pure supervised learning setting.""",
        'paper_link': None,
        'paper_name': None,
        'catch_prob': 'Catch trial probability. (def: 0.1)',
        'stim_th': '''Percentile of stimulus distribution below which catch
        trials are allowed (in some cases, experimenter might decide not
        to have catch trials when  stimulus is very obvious) (def: 50)''',
        'start': '''Number of trials after which the catch trials can occur.
        (def: 0)'''
    }

    def __init__(self, env, catch_prob=0.1, stim_th=50, start=0):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.catch_prob = catch_prob
        if stim_th is not None:
            self.stim_th = np.percentile(self.task.cohs, stim_th)
        else:
            self.stim_th = None
        self.R_CORRECT_ORI = self.task.R_CORRECT
        self.catch_trial = False
        # number of trials after which the prob. of catch trials is != 0
        self.start = start

    def new_trial(self, **kwargs):
        self.task.R_CORRECT = self.R_CORRECT_ORI
        coh = self.task.rng.choice(self.task.cohs)
        if self.stim_th is not None:
            if coh <= self.stim_th:
                self.catch_trial = self.task.rng.random() < self.catch_prob
            else:
                self.catch_trial = False
        else:
            self.catch_trial = self.task.rng.random() < self.catch_prob
        if self.catch_trial:
            self.task.R_CORRECT = self.task.R_FAIL
        kwargs.update({'coh': coh})
        self.env.new_trial(**kwargs)

    def _step(self, action):
        return self.env._step(action)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            if self.task.num_tr > self.start:
                info['catch_trial'] = self.catch_trial
                self.new_trial()
            else:
                info['catch_trial'] = False
        return obs, reward, done, info

    def seed(self, seed=None):
        self.task.seed(seed=seed)
        # keeps track of the repeating prob of the current block
        self.curr_block = self.task.rng.choice([0, 1])


if __name__ == '__main__':
    env = nalt_rdm.nalt_RDM(timing=[100, 200, 200, 200, 100], n=10)
    env = trial_hist_nalt.TrialHistory_NAlt(env)
    env = CatchTrials(env, catch_prob=0.7, stim_th=100)
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
    print(np.sum(np.array(rewards) == 1) / np.sum(np.argmax(gt, axis=1) == 1))

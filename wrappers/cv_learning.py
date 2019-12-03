#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:45:33 2019

@author: molano
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
from os.path import expanduser
from gym.core import Wrapper
from neurogym.envs import delayresponse as DR
from copy import copy
home = expanduser("~")
sys.path.append(home)
sys.path.append(home + '/neurogym')
sys.path.append(home + '/gym')


class CurriculumLearning(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, perf_w=10, max_num_reps=3, init_ph=0):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        self.ori_task = copy(env)
        self.curr_ph = init_ph
        self.curr_perf = 0
        self.perf_window = perf_w
        self.goal_perf = 0.8
        self.mov_window = np.repeat(0, 10)
        self.counter = 0
        self.max_num_reps = max_num_reps
        self._set_trial_params()
        self.task.trial = self.task._new_trial()
        self.rew = 0

    def _set_trial_params(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            self.task.stimulus_mean = 0
            self.task.stimulus_min = 0
            self.task.stimulus_max = 0
            self.task.decision = 1000000
            self.task.delays = [0]
            self.task.R_FAIL = self.task.R_CORRECT
            self.task.sigma = 0
            assert self.ori_task.R_FAIL != self.task.R_CORRECT, 'do a copy'
        elif self.curr_ph == 1:
            # there is stim but first answer is not penalized
            self.task.stimulus_min = self.ori_task.stimulus_min
            self.task.stimulus_mean = self.ori_task.stimulus_mean
            self.task.stimulus_max = self.ori_task.stimulus_max
            # self.task.decision = self.ori_task.decision
            self.task.R_FAIL = 0
            self.task.firstcounts = False
            self.task.cohs = np.array([100])
        elif self.curr_ph == 2:
            # first answer counts
            self.task.R_FAIL = self.ori_task.R_FAIL
            self.task.firstcounts = True
        elif self.curr_ph == 3:
            self.task.delays = self.ori_task.delays          
        elif self.curr_ph == 4:
            self.task.coh = self.ori_task.cohs
            self.task.sigma = self.ori_task.sigma

    def count(self, action):
        # analyzes the last three answers during stage 0
        new = self.task.actions[action]
        if np.sign(self.counter) == np.sign(new):
            self.counter += new
        else:
            self.counter = new

    def set_phase(self):
        if self.rew == self.task.R_CORRECT:
            self.mov_window = np.append(self.mov_window, 1)
            self.mov_window = np.delete(self.mov_window, 0)
        else:
            self.mov_window = np.append(self.mov_window, 0)
            self.mov_window = np.delete(self.mov_window, 0)
        if np.sum(self.mov_window)/self.perf_window >= self.goal_perf:
            self.curr_ph += 1
            self.mov_window = np.repeat(0, self.perf_window)

    def reset(self):
        return self.task.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)
        self.rew = reward
        if info['new_trial']:
            self.set_phase()
            self._set_trial_params()
            self.task.trial = self.task._new_trial()
            if self.curr_ph == 0:
                self.count(action)
                if np.abs(self.counter) == self.max_num_reps:
                    self.task.trial['ground_truth'] = -1 if action == 2 else 1
                    self.task.R_FAIL = self.ori_task.R_FAIL #or equal 0?
                    self.task.firstcounts = False
                    self.counter = 0
                elif np.abs(self.counter) == 1:
                    self.task.R_FAIL = self.ori_task.R_CORRECT
                    self.task.firstcounts = True
        g_t = self.task.trial['ground_truth']

        return obs, reward, done, info, g_t


if __name__ == '__main__':
    plt.close('all')
    rows = 3
    env = DR.DR(timing=[100, 300, 300, 300, 300])
    env = CurriculumLearning(env)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    num_steps_env = 300
    g_t = 0
    for stp in range(int(num_steps_env)):
        action = env.action_space.sample()
        action = int(g_t/2+1.5)
        print('gt', g_t)
        obs, rew, done, info, g_t = env.step(action)
        print(info['gt'])
        print(action)
        print(rew)
        print(obs)
        print('')
        if done:
            env.reset()
        observations.append(obs)
        if info['new_trial']:
            print('XXXXXXXXXXXX')
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
    observations = np.array(observations)
    plt.figure()
    plt.subplot(rows, 1, 1)
    plt.imshow(observations.T, aspect='auto')
    plt.title('observations')
    plt.subplot(rows, 1, 2)
    plt.plot(actions, marker='+')
    plt.plot(actions_end_of_trial, '--')
    gt = np.array(gt)
    print(np.argmax(gt[len(gt)-1]))
    #print(np.argmax(gt[0], axis=1))
    plt.plot(np.argmax(gt, axis=1), 'r')
    print(np.sum(np.argmax(gt, axis=1) == 2))
    print(np.sum(np.argmax(gt, axis=1) == 1))
    # aux = np.argmax(obs, axis=1)
    # aux[np.sum(obs, axis=1) == 0] = -1
    # plt.plot(aux, '--k')
    plt.title('actions')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.subplot(rows, 1, 3)
    plt.plot(rewards, 'r')
    plt.title('reward')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.show()
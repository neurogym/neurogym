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
from neurogym.envs import generaltask as GT
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
        self.goal_perf = [0.8, 0.8, 0.8, 0.8]
        self.mov_window = [0]*self.perf_window
        self.counter = 0
        self.max_num_reps = max_num_reps
        self.rew = 0
        self.new_trial()

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.first_trial_rew = None
        # self.set_phase()
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            if np.abs(self.counter) >= self.max_num_reps:
                ground_truth = 1 if self.action == 2 else 2
                kwargs.update({'gt': ground_truth})
                self.task.R_FAIL = 0
            else:
                self.task.R_FAIL = self.ori_task.R_CORRECT
            kwargs.update({'durs': {'stimulus': 0,
                                    'delay_aft_stim': 0},
                           'sigma': 0})
        elif self.curr_ph == 1:
            # stim introduced with no ambiguity
            # wrong answer is not penalized
            # agent can keep exploring until finding the right answer
            kwargs.update({'durs': {'delay_aft_stim': 0},
                           'cohs': np.array([100]), 'sigma': 0})
            self.task.R_FAIL = 0
            self.task.firstcounts = False
        elif self.curr_ph == 2:
            # first answer counts
            # wrong answer is penalized
            self.first_trial_rew = None
            self.task.R_FAIL = self.ori_task.R_FAIL
            self.task.firstcounts = True
            kwargs.update({'durs': {'delay_aft_stim': 0},
                           'cohs': np.array([100]), 'sigma': 0})
        elif self.curr_ph == 3:
            # delay component is introduced
            kwargs.update({'cohs': np.array([100]), 'sigma': 0})

        # phase 4: ambiguity component is introduced
        self.env.new_trial(**kwargs)

    def count(self, action):
        '''
        check the last three answers during stage 0 so the network has to
        alternate between left and right
        '''
        if action != 0:
            new = action - 2/action
            if np.sign(self.counter) == np.sign(new):
                self.counter += new
            else:
                self.counter = new
            # print('counter', self.counter)

    def set_phase(self):
        self.mov_window.append(1*(self.rew == self.task.R_CORRECT))
        self.mov_window.pop(0)  # remove first value
        self.curr_perf = np.sum(self.mov_window)/self.perf_window
        if self.curr_ph < 4 and self.curr_perf >= self.goal_perf[self.curr_ph]:
            self.curr_ph += 1
            self.mov_window = [0]*self.perf_window

    def reset(self):
        return self.task.reset()

    def _step(self, action):
        obs, reward, done, info = self.env._step(action)
        if ~self.env.firstcounts and ~np.isnan(info['first_trial']):
            self.first_trial_rew = reward
        self.rew = self.first_trial_rew or reward
        self.action = action
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.set_phase()
            info.update({'curr_ph': self.curr_ph})
            self.count(action)
            self.new_trial()

        return obs, reward, done, info


if __name__ == '__main__':
    plt.close('all')
    rows = 3
    timing = {'fixation': [200, 200, 200], 'stimulus': [200, 100, 300],
              'delay_btw_stim': [0, 0, 0],
              'delay_aft_stim': [500, 200, 800], 'decision': [200, 200, 200]}
    simultaneous_stim = True
    env = GT.GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    env = CurriculumLearning(env)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    perf = []
    num_steps_env = 2000
    g_t = 0
    next_ph = 1
    for stp in range(int(num_steps_env)):
        action = env.gt
        # action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
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
            print('Current phase: ', info['curr_ph'])
            actions_end_of_trial.append(action)
            perf.append(env.curr_perf)
            if info['curr_ph'] == next_ph:
                plt.figure()
                plt.plot(perf)
                plt.title('Performance along phase ' + str(next_ph-1))
                plt.show()
                next_ph += 1
                perf = []
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
    # print(np.argmax(gt[0], axis=1))
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

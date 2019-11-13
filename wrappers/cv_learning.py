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
home = expanduser("~")
sys.path.append(home)
sys.path.append(home + '/neurogym')
sys.path.append(home + '/gym')
from gym.core import Wrapper
from neurogym.envs import delayresponse as DR


class CurriculumLearning(Wrapper):
    """
    modfies a given environment by changing the probability of repeating the
    previous correct response
    """
    def __init__(self, env, perf_w=10, max_num_reps=2):
        Wrapper.__init__(self, env=env)
        self.env = env
        # we get the original task, in case we are composing wrappers
        env_aux = env
        while env_aux.__class__.__module__.find('wrapper') != -1:
            env_aux = env.env
        self.task = env_aux
        #        self.ori_task = {}
        #        self.ori_task = type('ori_task',
        #                             self.task.__bases__, dict(self.task.__dict__))
        self.curr_ph = 0
        self.curr_perf = 0
        self.perf_window = perf_w
        self.counter = 0
        self.max_num_reps = max_num_reps

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
            # self.task.delays
            print(self.ori_task.R_FAIL)
            self.task.R_FAIL = self.task.R_CORRECT
            print(self.ori_task.R_FAIL)
            assert self.ori_task.R_FAIL != self.task.R_CORRECT, 'do a copy'
        elif self.curr_ph == 1:
            # there is stim but first answer is not penalized
            self.task.stimulus_min = self.ori_task.stimulus_min
            self.task.stimulus_mean = self.ori_task.stimulus_mean
            self.task.stimulus_max = self.ori_task.stimulus_max
            self.task.decision = self.ori_task.decision
            self.task.R_FAIL = 0
            self.task.firstcounts = False
            self.task.cohs = np.array([100])
        elif self.curr_ph == 2:
            # first answer counts
            # TODO: use self.ori_task to recover original values for all
            # variables modified in phase 1
            self.task.R_FAIL = -self.task.R_CORRECT
            self.task.firstcounts = True
        elif self.curr_ph == 3:
            self.task.delays = [1000, 5000, 10000]
        elif self.curr_ph == 4:
            self.task.coh = np.array([0, 6.4, 12.8, 25.6, 51.2]) *\
                self.task.stimEv

    def count(self, action):
        # analyzes the last three answers during stage 0
        # self.alternate = False
        new = self.task.actions[action]
        if np.sign(self.counter) == np.sign(new):
            self.counter += new
        else:
            self.counter = new

    def reset(self):
        return self.task.reset()

    def step(self, action):
        obs, reward, done, info = self.env._step(action)
        if info['new_trial']:
            self._set_trial_params()
            self.task.trial = self.task._new_trial()
            if self.curr_ph == 0:
                self.count(action)
                if np.abs(self.counter) >= self.max_num_reps:
                    self.task.trial['ground_truth'] = 1 if action == 2 else 2
                    self.task.R_FAIL = self.ori_task.R_FAIL
                    self.task.firstcounts = False
                    self.counter = 0
                elif np.abs(self.counter) == 1:
                    self.task.R_FAIL = self.task.R_CORRECT
                    self.task.firstcounts = True

        return obs, reward, done, info


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
    num_steps_env = 200
    for stp in range(int(num_steps_env)):
        action = env.action_space.sample()
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


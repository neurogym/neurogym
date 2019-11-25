#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:55:24 2019

@author: martafradera
"""

from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
from neurogym.envs import ngym
import matplotlib.pyplot as plt


class DR(ngym.ngym):
    def __init__(self, dt=100, timing=[],
                 stimEv=1., **kwargs):
        super().__init__(dt=dt)
        # Actions (fixate, left, right)
        self.actions = [0, -1, 1]
        # trial conditions (left, right)
        self.choices = [-1, 1]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.fixation = timing[0]
        self.stimulus_min = timing[1]
        self.stimulus_mean = timing[2]
        self.stimulus_max = timing[3]
        self.decision = timing[4]
        self.delays = [1000, 5000, 10000]
        self.mean_trial_duration = self.fixation + self.stimulus_mean +\
            np.mean(self.delays) + self.decision
        if self.fixation == 0 or self.decision == 0 or self.stimulus_mean == 0:
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXX')
        print('Tiffanys Task')
        print('Fixation: ' + str(self.fixation))
        print('Min Stimulus Duration: ' + str(self.stimulus_min))
        print('Mean Stimulus Duration: ' + str(self.stimulus_mean))
        print('Max Stimulus Duration: ' + str(self.stimulus_max))
        print('Delay: ' + str(self.delays))
        print('Decision: ' + str(self.decision))
        print('(time step: ' + str(self.dt) + ')')
        print('Mean Trial Duration: ' + str(self.mean_trial_duration))
        print('XXXXXXXXXXXXXXXXXXXXXX')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -1.
        self.R_MISS = 0.
        self.abort = False
        self.firstcounts = True
        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        # start new trial
        self.trial = self._new_trial()

    def _new_trial(self):
        """
        _new_trial() is called when a trial ends to get the specifications of
        the next trial. Such specifications are stored in a dictionary with
        the following items:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial

        """

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)

        self.delay = self.rng.choice(self.delays)

        # maximum length of current trial
        self.tmax = self.fixation + stimulus + self.delay + self.decision
        durations = {
            'fixation': (0, self.fixation),
            'stimulus': (self.fixation, self.fixation + stimulus),
            'delay': (self.fixation + stimulus,
                      self.fixation + stimulus + self.delay),
            'decision': (self.fixation + stimulus + self.delay,
                         self.fixation + stimulus + self.delay +
                         self.decision),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice(self.choices)
        coh = self.rng.choice(self.cohs)

        return {
            'durations': durations,
            'ground_truth': ground_truth,
            'coh': coh
            }

        # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        info['gt'] = np.zeros((3,))
        # rewards
        reward = 0
        # observations
        obs = np.zeros((3,))
        if self.in_epoch(self.t, 'fixation'):
            info['gt'][0] = 1
            obs[0] = 1
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED

        elif self.in_epoch(self.t, 'delay'):
            info['gt'][0] = 1
            if self.actions[action] != 0:
                reward = self.R_ABORTED
                info['new_trial'] = self.abort

        elif self.in_epoch(self.t, 'decision'):
            info['gt'][int((trial['ground_truth']/2+1.5))] = 1
            gt_sign = np.sign(trial['ground_truth'])
            action_sign = np.sign(self.actions[action])
            if gt_sign == action_sign:
                reward = self.R_CORRECT
                info['new_trial'] = True
            elif gt_sign == -action_sign:
                reward = self.R_FAIL
                info['new_trial'] = self.firstcounts


        else:
            info['gt'][0] = 1

        # this is an 'if' to allow the stimulus and fixation periods to overlap
        if self.in_epoch(self.t, 'stimulus'):
            obs[0] = 1
            high = (trial['ground_truth'] > 0) + 1
            low = (trial['ground_truth'] < 0) + 1
            obs[high] = self.scale(trial['coh']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, info['new_trial'] = tasktools.new_trial(self.t, self.tmax,
                                                        self.dt,
                                                        info['new_trial'],
                                                        self.R_MISS, reward)
        if info['new_trial']:
            self.t = 0
            self.num_tr += 1
        else:
            self.t += self.dt
            print(self.t)

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        """
        step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        Note that the main computations are done by the function _step(action),
        and the extra lines are basically checking whether to call the
        _new_trial() function in order to start a new trial
        """
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info


if __name__ == '__main__':
    plt.close('all')
    rows = 3
    env = DR(timing=[100, 300, 300, 300, 300])
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

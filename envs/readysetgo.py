#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Ready-Set-Go task and Contextual Ready-Set-Go task, based on

  Flexible Sensorimotor Computations through Rapid
  Reconfiguration of Cortical Dynamics
  Evan D. Remington, Devika Narain,
  Eghbal A. Hosseini, Mehrdad Jazayeri, Neuron 2018.

  https://www.cell.com/neuron/pdf/S0896-6273(18)30418-5.pdf

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
from neurogym.envs import ngym


class ReadySetGo(ngym.ngym):
    def __init__(self, dt=80, timing=(500, 83, 83), gain=1):
        super().__init__(dt=dt)
        if dt > 80:
            raise ValueError('dt {:0.2f} too large for this task.'.format(dt))
        # Actions (fixate, go)
        self.actions = [-1, 1]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        # possible durations
        self.measures = [500, 580, 660, 760, 840, 920, 1000]
        # gain
        self.gain = gain

        # Durations
        self.fixation = timing[0]  # 500
        self.ready = timing[1]  # 83
        self.set = timing[2]  # 83
        max_trial_duration = self.fixation + np.max(self.measures) +\
            self.set + 2*gain*self.set
        if self.fixation == 0 or self.ready == 0 or self.set == 0:
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('mean trial duration: ' + str(max_trial_duration) +
              ' (max num. steps: ' +
              str(max_trial_duration/self.dt) + ')')

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        measure = tasktools.choice(self.rng, self.measures)
        production = measure * self.gain
        self.tmax = self.fixation + measure + self.set + 2*production

        durations = {
            'fixation':  (0, self.fixation),
            'ready': (self.fixation, self.fixation + self.ready),
            'measure': (self.fixation, self.fixation + measure),
            'set': (self.fixation + measure,
                    self.fixation + measure + self.set),
            'production': (self.fixation + measure + self.set,
                           self.tmax),
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        return {
            'durations': durations,
            'measure': measure,
            'production': production,
            }

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False, 'gt': np.zeros((2,))}
        reward = 0
        obs = np.zeros((3,))
        if self.in_epoch(self.t, 'fixation'):
            obs[0] = 1
            if self.actions[action] != -1:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'production'):
            t_prod = self.t - trial['durations']['measure'][1]
            eps = abs(t_prod - trial['production'])
            if eps < self.dt/2 + 1:
                info['gt'][1] = 1
            else:
                info['gt'][0] = 1
            if action == 1:
                info['new_trial'] = True  # terminate
                # actual production time
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    reward = self.R_FAIL
                else:
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT
        else:
            info['gt'][0] = 1

        if self.in_epoch(self.t, 'ready'):
            obs[1] = 1
        if self.in_epoch(self.t, 'set'):
            obs[2] = 1

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

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info

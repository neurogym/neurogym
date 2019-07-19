# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:00:32 2019

@author: MOLANO

A parametric working memory task, based on

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://dx.doi.org/10.1523/JNEUROSCI.1875-10.2010

"""
import numpy as np
from gym import spaces
from neurogym.ops import tasktools
from neurogym.envs import ngym


class Romo(ngym.ngym):
    def __init__(self, dt=100, timing=(750, 500, 2700, 3300, 500, 500)):
        # call ngm __init__ function
        super().__init__(dt=dt)

        # Actions (fixate, left, right)
        self.actions = [0, -1, 1]

        # trial conditions
        self.choices = [-1, 1]
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

        # Input noise
        self.sigma = np.sqrt(2*100*0.001)

        # Epoch durations
        self.fixation = timing[0]
        self.f1 = timing[1]
        self.delay_min = timing[2]  # 3000 - 300
        self.delay_max = timing[3]  # 3000 + 300
        self.delay_mean = (self.delay_max + self.delay_min) / 2
        self.f2 = timing[4]
        self.decision = timing[5]
        self.mean_trial_duration = self.fixation + self.f1 + self.delay_mean +\
            self.f2 + self.decision
        if self.fixation == 0 or self.decision == 0 or self.delay_mean == 0:
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' +
              str(self.mean_trial_duration/self.dt) + ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        # Input scaling
        self.fall = np.ravel(self.fpairs)
        self.fmin = np.min(self.fall)
        self.fmax = np.max(self.fall)

        # action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2, ),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        self.trial = self._new_trial()

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)
        self.tmax = self.fixation + self.f1 + delay + self.f2 + self.decision
        durations = {
            'fixation':   (0, self.fixation),
            'f1':         (self.fixation, self.fixation + self.f1),
            'delay':      (self.fixation + self.f1,
                           self.fixation + self.f1 + delay),
            'f2':         (self.fixation + self.f1 + delay,
                           self.fixation + self.f1 + delay + self.f2),
            'decision':   (self.fixation + self.f1 + delay + self.f2,
                           self.tmax),
            }

        ground_truth = tasktools.choice(self.rng, self.choices)
        fpair = tasktools.choice(self.rng, self.fpairs)
        if ground_truth == -1:
            f1, f2 = fpair
        else:
            f2, f1 = fpair
        return {
            'durations': durations,
            'ground_truth': ground_truth,
            'f1': f1,
            'f2': f2
            }

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        # rewards
        reward = 0
        # observations
        obs = np.zeros((2,))
        if self.in_epoch(self.t, 'fixation'):
            obs[0] = 1
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch(self.t, 'decision'):
            gt_sign = np.sign(trial['ground_truth'])
            action_sign = np.sign(self.actions[action])
            if gt_sign == action_sign:
                reward = self.R_CORRECT
            elif gt_sign == -action_sign:
                reward = self.R_FAIL
            info['new_trial'] = self.actions[action] != 0

        if self.in_epoch(self.t, 'f1'):
            obs[1] = self.scale_p(trial['f1']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
        elif self.in_epoch(self.t, 'f2'):
            obs[1] = self.scale_p(trial['f2']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, info['new_trial'] = tasktools.new_trial(self.t, self.tmax,
                                                        self.dt,
                                                        info['new_trial'],
                                                        self.R_MISS, reward)
        info['gt'] = np.zeros((3,))
        if info['new_trial']:
            info['gt'][int((trial['ground_truth']/2+1.5))] = 1
            self.t = 0
            self.num_tr += 1
        else:
            self.t += self.dt
            info['gt'][0] = 1

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.trial = self._new_trial()
        return obs, reward, done, info

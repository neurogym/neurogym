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
import neurogym as ngym


class Romo(ngym.EpochEnv):
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

    def __str__(self):
        string = ''
        if self.fixation == 0 or self.decision == 0 or self.delay_mean == 0:
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += ' (max num. steps: ' + str(self.mean_trial_duration/self.dt)
        return string

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)

        self.add_epoch('fixation', self.fixation, start=0)
        self.add_epoch('f1', self.f1, after='fixation')
        self.add_epoch('delay', delay, after='f1')
        self.add_epoch('f2', self.f2, after='delay')
        self.add_epoch('decision', self.decision, after='f2', last_epoch=True)

        ground_truth = self.rng.choice(self.choices)
        fpair = self.rng.choice(self.fpairs)
        if ground_truth == -1:
            f1, f2 = fpair
        else:
            f2, f1 = fpair
        return {
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
        info['gt'] = np.zeros((3,))
        if self.in_epoch('fixation'):
            info['gt'][0] = 1
            obs[0] = 1
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            info['gt'][int((trial['ground_truth']/2+1.5))] = 1
            gt_sign = np.sign(trial['ground_truth'])
            action_sign = np.sign(self.actions[action])
            if gt_sign == action_sign:
                reward = self.R_CORRECT
            elif gt_sign == -action_sign:
                reward = self.R_FAIL
            info['new_trial'] = self.actions[action] != 0
        else:
            info['gt'][0] = 1

        if self.in_epoch('f1'):
            obs[1] = self.scale_p(trial['f1']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)
        elif self.in_epoch('f2'):
            obs[1] = self.scale_p(trial['f2']) +\
                self.rng.gauss(mu=0, sigma=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, info['new_trial'] = tasktools.new_trial(self.t, self.tmax,
                                                        self.dt,
                                                        info['new_trial'],
                                                        self.R_MISS, reward)

        return obs, reward, False, info

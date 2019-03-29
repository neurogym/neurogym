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
import tasktools
import ngym
from gym import spaces


class Romo(ngym.ngym):
    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)
        # Inputs
        self.inputs = tasktools.to_map('FIXATION', 'F-POS', 'F-NEG')

        # Actions
        self.actions = tasktools.to_map('FIXATE', '>', '<')

        # trial conditions
        self.choices = ['>', '<']
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

        # Input noise
        self.sigma = np.sqrt(2*100*0.001)

        # Epoch durations
        self.fixation = 750
        self.f1 = 500
        self.delay_min = 3000 - 300
        self.delay_max = 3000 + 300
        self.delay_mean = (self.delay_max + self.delay_min) / 2
        self.f2 = 500
        self.decision = 500
        self.mean_trial_duration = self.fixation + self.f1 + self.delay_mean +\
            self.f2 + self.decision
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.abort = False

        # Input scaling
        self.fall = np.ravel(self.fpairs)
        self.fmin = np.min(self.fall)
        self.fmax = np.max(self.fall)
        # action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
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

        return {
            'durations': durations,
            'ground_truth': ground_truth,
            'fpair': fpair
            }

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def _step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        # epochs = trial['epochs']
        info = {'new_trial': False}
        reward = 0
        tr_perf = False
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['FIXATE']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action == self.actions['>']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] == '>'):
                    reward = self.R_CORRECT
            elif action == self.actions['<']:
                tr_perf = True
                info['new_trial'] = True
                if (trial['ground_truth'] == '<'):
                    reward = self.R_CORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        if trial['ground_truth'] == '>':
            f1, f2 = trial['fpair']
        else:
            f2, f1 = trial['fpair']

        obs = np.zeros(len(self.inputs))
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'f1'):
            obs[self.inputs['F-POS']] = self.scale_p(f1) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[self.inputs['F-NEG']] = self.scale_n(f1) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
        if self.in_epoch(self.t, 'f2'):
            obs[self.inputs['F-POS']] = self.scale_p(f2) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)
            obs[self.inputs['F-NEG']] = self.scale_n(f2) +\
                self.rng.normal(scale=self.sigma)/np.sqrt(self.dt)

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['new_trial'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            info['gt'] = trial['ground_truth']
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf,
                                       tr_perf)
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

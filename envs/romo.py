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

    # Inputs
    inputs = tasktools.to_map('FIXATION', 'F-POS', 'F-NEG')

    # Actions
    actions = tasktools.to_map('FIXATE', '>', '<')

    # Trial conditions
    ground_truth = ['>', '<']
    fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

    # Input noise
    sigma = np.sqrt(2*100*0.001)

    # Epoch durations
    # TODO: in ms?
    fixation = 750
    f1 = 500
    delay_min = 3000 - 300
    delay_max = 3000 + 300
    f2 = 500
    decision = 500
    tmax = fixation + f1 + delay_max + f2 + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_MISS = 0.
    abort = False

    # Input scaling
    fall = np.ravel(fpairs)
    fmin = np.min(fall)
    fmax = np.max(fall)

    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.num_tr_exp = 1000

        self.trial = self._new_trial()

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)

        durations = {
            'fix_grace': (0, 100),
            'fixation':   (0, self.fixation),
            'f1':         (self.fixation, self.fixation + self.f1),
            'delay':      (self.fixation + self.f1,
                           self.fixation + self.f1 + delay),
            'f2':         (self.fixation + self.f1 + delay,
                           self.fixation + self.f1 + delay + self.f2),
            'decision':   (self.fixation + self.f1 + delay + self.f2,
                           self.tmax),
            'tmax':       self.tmax
            }

        ground_truth = self.rng.choice(self.ground_truth)

        fpair = self.rng.choice(self.fpairs)

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

    def step(self, action):
        trial = self.trial

        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        # epochs = trial['epochs']
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if not self.in_epoch(self.t - 1, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                info['choice'] = None
                reward = self.R_ABORTED
        else:
            if action == self.actions['>']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = '>'
                info['correct'] = (trial['ground_truth'] == '>')
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['<']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = '<'
                info['correct'] = (trial['ground_truth'] == '<')
                if info['correct']:
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
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt

        self.store_data(obs, action, reward)
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.97

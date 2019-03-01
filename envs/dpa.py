#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:25:08 2019

@author: molano


Delay Pair Association (DPA) task based on:

  Active information maintenance in working memory by a sensory cortex
  Xiaoxing Zhang, Wenjun Yan, Wenliang Wang, Hongmei Fan, Ruiqing Hou,
  Yulei Chen, Zhaoqin Chen, Shumin Duan, Albert Compte, Chengyu Li bioRxiv 2018

  https://www.biorxiv.org/content/10.1101/385393v1

"""
import numpy as np
import tasktools
import ngym
from gym import spaces


class DPA(ngym.ngym):

    # Inputs
    inputs = tasktools.to_map('FIXATION', 'S1', 'S2', 'S3', 'S4')

    # Actions
    # TODO: fixate != nogo?
    actions = tasktools.to_map('FIXATE', 'NO_MATCH', 'MATCH')

    # Trial conditions
    fnew_trial = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Input noise
    sigma = np.sqrt(2*100*0.001)

    # Epoch durations
    fixation = 500
    dpa1 = 1000
    delay_min = 13000
    delay_max = 13001
    dpa2 = 1000
    resp_delay = 1000
    decision = 500
    tmax = fixation + dpa1 + delay_max + dpa2 + resp_delay + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_MISS = 0.

    def __init__(self, dt=500):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        # hight = np.array([1.])
        self.observation_space = spaces.Box(-1., 1, shape=(5, ),
                                            dtype=np.float32)
        # TODO: are these necessary?
        self.seed()
        self.viewer = None

        self.steps_beyond_done = None
        ###################

        self.trial = self._new_trial()

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Epochs
        # --------------------------------------------------------------------------

        delay = tasktools.uniform(self.rng, self.dt, self.delay_min,
                                  self.delay_max)

        durations = {
            'fixation_grace': (0, 100),
            'fixation':   (0, self.fixation),
            'dpa1':         (self.fixation, self.fixation + self.dpa1),
            'delay':      (self.fixation + self.dpa1,
                           self.fixation + self.dpa1 + delay),
            'dpa2':         (self.fixation + self.dpa1 + delay,
                             self.fixation + self.dpa1 + delay + self.dpa2),
            'resp_delay': (self.fixation + self.dpa1 + delay + self.dpa2,
                           self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay),
            'decision':   (self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay, self.tmax),
            'tmax':       self.tmax
            }

        pair = tasktools.choice(self.rng, self.fnew_trial)

        if np.diff(pair)[0] == 0:
            ground_truth = 'MATCH'
        else:
            ground_truth = 'NO_MATCH'

        return {
            'durations': durations,
            'ground_truth':     ground_truth,
            'pair':     pair
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
        # if self.t not in epochs['decision']:
        if not self.in_epoch(self.t, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fixation_grace')):
                info['continue'] = False
                info['choice'] = None
                reward = self.R_ABORTED
        else:  # elif self.t in epochs['decision']:
            if action == self.actions['NO_MATCH']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'NO_MATCH'
                info['correct'] = (trial['ground_truth'] == 'NO_MATCH')
                if info['correct']:
                    reward = self.R_CORRECT
            elif action == self.actions['MATCH']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'MATCH'
                info['correct'] = (trial['ground_truth'] == 'MATCH')
                if info['correct']:
                    reward = self.R_CORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        dpa1, dpa2 = trial['pair']
        obs = np.zeros(len(self.inputs))
        # if self.t not in epochs['decision']:
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        # if self.t in epochs['dpa1']:
        if self.in_epoch(self.t, 'dpa1'):
            # TODO: there is a more efficient way to do this,
            # without using self.inputs. Do we need self.inputs at al?
            if dpa1 == 0:
                obs[self.inputs['S1']] = 1
            elif dpa1 == 1:
                obs[self.inputs['S2']] = 1
        # if self.t in epochs['dpa2']:
        if self.in_epoch(self.t, 'dpa2'):
            if dpa2 == 0:
                obs[self.inputs['S3']] = 1
            elif dpa2 == 1:
                obs[self.inputs['S4']] = 1

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

        done = False
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.97

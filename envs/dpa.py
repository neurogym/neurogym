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
    actions = tasktools.to_map('NO_GO', 'GO')

    # trial conditions
    dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]

    # Input noise
    sigma = np.sqrt(2*100*0.001)

    # Epoch durations
    fixation = 0
    dpa1 = 1000
    delay_min = 1000  # Original paper: 13000
    delay_max = 1001
    dpa2 = 1000
    resp_delay = 1000
    decision = 500
    tmax = fixation + dpa1 + delay_max + dpa2 + resp_delay + decision

    # Rewards
    R_ABORTED = -0.1
    R_CORRECT = +1.
    R_INCORRECT = -1.
    R_MISS = 0.
    abort = False

    def __init__(self, dt=100):
        # call ngm __init__ function
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(2)
        # hight = np.array([1.])
        self.observation_space = spaces.Box(-1., 1, shape=(5, ),
                                            dtype=np.float32)
        # TODO: are these necessary?
        self.seed()
        self.viewer = None

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
            'dpa1':         (self.fixation, self.fixation + self.dpa1),
            'delay':      (self.fixation + self.dpa1,
                           self.fixation + self.dpa1 + delay),
            'dpa2':         (self.fixation + self.dpa1 + delay,
                             self.fixation + self.dpa1 + delay + self.dpa2),
            'resp_delay': (self.fixation + self.dpa1 + delay + self.dpa2,
                           self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay),
            'decision':   (self.fixation + self.dpa1 + delay + self.dpa2 +
                           self.resp_delay, self.fixation + self.dpa1 +
                           delay + self.dpa2 + self.resp_delay +
                           self.decision),
            'tmax':       self.tmax
            }

        pair = tasktools.choice(self.rng, self.dpa_pairs)

        if np.diff(pair)[0] == 2:
            ground_truth = 'GO'
        else:
            ground_truth = 'NO_GO'

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

    def _step(self, action):
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
            if (action != self.actions['NO_GO'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                info['choice'] = None
                reward = self.R_ABORTED
        else:  # elif self.t in epochs['decision']:
            # print('decision period')
            if action == self.actions['GO']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = 'GO'
                info['correct'] = (trial['ground_truth'] == 'GO')
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_INCORRECT

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        dpa1, dpa2 = trial['pair']
        #        print('action: ' + str(action))
        #        print('match: ' + str(np.diff(trial['pair'])[0]))
        #        print('gt: ' + trial['ground_truth'])
        obs = np.zeros(len(self.inputs))
        # if self.t not in epochs['decision']:
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        # if self.t in epochs['dpa1']:
        if self.in_epoch(self.t, 'dpa1'):
            # without using self.inputs. Do we need self.inputs at al?
            obs[dpa1] = 1
        # if self.t in epochs['dpa2']:
        if self.in_epoch(self.t, 'dpa2'):
            obs[dpa2] = 1
        # ---------------------------------------------------------------------
        # new trial?
        dec_per_end = trial['durations']['decision'][1]
        reward, new_trial = tasktools.new_trial(self.t, dec_per_end, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            #            print('new trial')
            info['new_trial'] = True
            info['gt'] = trial['ground_truth']
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt
        #        print('reward: ' + str(reward))
        #        print('observation: ' + str(obs))
        #        print('---------------------')
        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.97

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to sample

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
from neurogym.envs import ngym


class DelayedMatchToSample(ngym.ngym):
    def __init__(self, dt=100, timing=(500, 500, 1500, 500, 500)):
        super().__init__(dt=dt)
        # TODO: Code a continuous space version
        # Actions ('FIXATE', 'MATCH', 'NONMATCH')
        self.actions = [0, -1, 1]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # TODO: Find these info from a paper
        self.fixation = timing[0]
        self.sample = timing[1]
        self.delay = timing[2]
        self.test = timing[3]
        self.decision = timing[4]
        self.tmax = np.sum(timing)

        mean_trial_duration = self.tmax
        if self.fixation == 0 or self.sample == 0 or self.delay == 0 or\
           self.test == 0 or self.decision == 0:
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('mean trial duration: ' + str(mean_trial_duration) +
              ' (max num. steps: ' + str(mean_trial_duration/self.dt) + ')')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        self.action_space = spaces.Discrete(3)
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
        dur = {'tmax': self.tmax}
        dur['fixation'] = (0, self.fixation)
        dur['sample'] = (dur['fixation'][1], dur['fixation'][1] + self.sample)
        dur['delay'] = (dur['sample'][1], dur['sample'][1] + self.delay)
        dur['test'] = (dur['delay'][1], dur['delay'][1] + self.test)
        dur['decision'] = (dur['test'][1], dur['test'][1] + self.decision)
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        # TODO: may need to fix this
        gt = tasktools.choice(self.rng, [-1, 1])
        sample = tasktools.choice(self.rng, [0, 1])
        if gt == 1:
            test = sample
        else:
            test = 1*(not sample)

        return {
            'durations': dur,
            'ground_truth': gt,
            'sample': sample,
            'test': test,
            }

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False, 'gt': np.zeros((3,))}
        reward = 0
        obs = np.zeros((3,))

        if self.in_epoch(self.t, 'fixation'):
            info['gt'][0] = 1
            obs[0] = 1
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch(self.t, 'decision'):
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

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        if self.in_epoch(self.t, 'sample'):
            obs[trial['sample']+1] = 1
        if self.in_epoch(self.t, 'test'):
            obs[trial['test']+1] = 1

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

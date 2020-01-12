#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to sample

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


class DelayedMatchToSample(ngym.EpochEnv):
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

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.trial = self._new_trial()

    def __str__(self):
        mean_trial_duration = self.tmax
        string = ''
        if self.fixation == 0 or self.sample == 0 or self.delay == 0 or\
           self.test == 0 or self.decision == 0:
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'mean trial duration: ' + str(mean_trial_duration)
        string += ' (max num. steps: ' + str(mean_trial_duration/self.dt) + '\n'
        return string

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice([1, 2])
        sample = self.rng.choice([1, 2])
        if ground_truth == 1:
            test = sample
        else:
            test = 3 - sample
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', self.fixation, start=0)
        self.add_epoch('sample', self.sample, after='fixation')
        self.add_epoch('delay', self.delay, after='sample')
        self.add_epoch('test', self.test, after='delay')
        self.add_epoch('decision', self.decision, after='test', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        tmp = [1, 0, 0]
        tmp[sample] = 1
        self.set_ob('sample', tmp)
        tmp = [1, 0, 0]
        tmp[test] = 1
        self.set_ob('delay', [1, 0, 0])
        self.set_ob('test', tmp)

        self.obs[self.sample_ind0:self.sample_ind1, 1:] += np.random.randn(
            self.sample_ind1-self.sample_ind0, 2) * (self.sigma/np.sqrt(self.dt))

        self.obs[self.test_ind0:self.test_ind1, 1:] += np.random.randn(
            self.test_ind1-self.test_ind0, 2) * (self.sigma/np.sqrt(self.dt))

        self.set_groundtruth('decision', ground_truth)

        return {
            'ground_truth': ground_truth,
            'sample': sample,
            'test': test,
            }

    def _step(self, action):
        info = {'new_trial': False}
        reward = 0

        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]

        if self.in_epoch('fixation'):
            if self.actions[action] != 0:
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                info['new_trial'] = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, info

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to sample

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class DelayedMatchToSample(ngym.ngym):
    def __init__(self, dt=100):
        super().__init__(dt=dt)
        # Inputs
        # TODO: Code a continuous space version
        self.inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

        # Actions
        self.actions = tasktools.to_map('FIXATE', 'MATCH', 'NONMATCH')

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)

        # TODO: Find these info from a paper
        self.tmax = 3500
        self.fixation = 500
        self.sample = 500
        self.delay = 1500
        self.test = 500
        self.decision = 500
        self.mean_trial_duration = self.tmax
        print('mean trial duration: ' + str(self.mean_trial_duration) +
              ' (max num. steps: ' + str(self.mean_trial_duration/self.dt) +
              ')')
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

        # TODO: this is a lot of repeated typing
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
        match = tasktools.choice(self.rng, ['MATCH', 'NONMATCH'])
        sample = tasktools.choice(self.rng, ['LEFT', 'RIGHT'])
        if match == 'MATCH':
            test = sample
        else:
            test = 'LEFT' if sample == 'RIGHT' else 'RIGHT'

        return {
            'durations': dur,
            'match': match,
            'sample': sample,
            'test': test,
            }

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False
        if self.in_epoch(self.t, 'fixation'):
            if (action != self.actions['FIXATE']):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        if self.in_epoch(self.t, 'decision'):
            if action != self.actions['FIXATE']:
                tr_perf = True
                info['continue'] = False
                info['choice'] = action
                info['t_choice'] = self.t
                info['correct'] = (action == self.actions[trial['match']])
                if info['correct']:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------
        obs = np.zeros(len(self.inputs))
        if not self.in_epoch(self.t, 'decision'):
            obs[self.inputs['FIXATION']] = 1
        if self.in_epoch(self.t, 'sample'):
            obs[self.inputs[trial['sample']]] = 1
        if self.in_epoch(self.t, 'test'):
            obs[self.inputs[trial['test']]] = 1

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax, self.dt,
                                                info['continue'],
                                                self.R_MISS, reward)

        if new_trial:
            info['new_trial'] = True
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward,
                                       self.num_tr_perf, tr_perf)
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info, new_trial

    def step(self, action):
        obs, reward, done, info, new_trial = self._step(action)
        if new_trial:
            self.trial = self._new_trial()
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8

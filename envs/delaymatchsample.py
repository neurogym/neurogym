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
    # Inputs
    # TODO: Code a continuous space version
    inputs = tasktools.to_map('FIXATION', 'LEFT', 'RIGHT')

    # Actions
    actions = tasktools.to_map('FIXATE', 'MATCH', 'NONMATCH')

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # TODO: Find these info from a paper
    tmax = 3500
    fixation = 500
    sample = 500
    delay = 1500
    test = 500
    decision = 500

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.
    abort = False

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.num_tr_exp = 1000

        self.trial = self._new_trial()
        print('------------------------')
        print('RDM task')
        print('time step: ' + str(self.dt))
        print('------------------------')

    def _new_trial(self):
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        # TODO: this is a lot of repeated typing
        dur = {'tmax': self.tmax}
        dur['fixation'] = (0, self.fixation)
        dur['fix_grace'] = (0, 100)
        dur['sample'] = (dur['fixation'][1], dur['fixation'][1] + self.sample)
        dur['delay'] = (dur['sample'][1], dur['sample'][1] + self.delay)
        dur['test'] = (dur['delay'][1], dur['delay'][1] + self.test)
        dur['decision'] = (dur['test'][1], dur['test'][1] + self.decision)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        # TODO: may need to fix this
        match = self.rng.choice(['MATCH', 'NONMATCH'])
        sample = self.rng.choice(['LEFT', 'RIGHT'])
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

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False
        # TODO: what happens if the network doesn't choose anything?
        # TODO: why is reward determined after input?
        if not self.in_epoch(self.t, 'decision'):
            if (action != self.actions['FIXATE'] and
                    not self.in_epoch(self.t, 'fix_grace')):
                info['continue'] = not self.abort
                reward = self.R_ABORTED
        else:
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
            self.t = 0
            self.num_tr += 1
            # compute perf
            self.perf, self.num_tr, self.num_tr_perf =\
                tasktools.compute_perf(self.perf, reward, self.num_tr,
                                       self.p_stp, self.num_tr_perf, tr_perf)
            self.trial = self._new_trial()
        else:
            self.t += self.dt

        done = self.num_tr > self.num_tr_exp
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8

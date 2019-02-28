#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:48:19 2019

@author: molano


Ready-Set-Go task and Contextual Ready-Set-Go task, based on

  Flexible Sensorimotor Computations through Rapid
  Reconfiguration of Cortical Dynamics
  Evan D. Remington, Devika Narain,
  Eghbal A. Hosseini, Mehrdad Jazayeri, Neuron 2018.

  https://www.cell.com/neuron/pdf/S0896-6273(18)30418-5.pdf

"""
from __future__ import division

import numpy as np
from gym import spaces
import tasktools
import ngym


class ReadySetGo(ngym.ngym):
    # Inputs
    inputs = tasktools.to_map('FIXATION', 'READY', 'SET')

    # Actions
    actions = tasktools.to_map('FIXATE', 'GO')

    # Input noise
    sigma = np.sqrt(2*100*0.01)

    # Durations
    fixation = 750
    stimulus_min = 80
    stimulus_mean = 330
    stimulus_max = 1500
    decision = 500
    tmax = fixation + stimulus_max + decision

    # Rewards
    R_ABORTED = -1.
    R_CORRECT = +1.
    R_FAIL = 0.
    R_MISS = 0.

    def __init__(self, dt=100):
        super().__init__(dt=dt)
        if dt > 80:
            raise ValueError('dt {:0.2f} too large for this task.'.format(dt))
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3, ),
                                            dtype=np.float32)

        self.seed()
        self.viewer = None

        self.steps_beyond_done = None

        self.trial = self._new_trial(self.rng, self.dt)
        print('------------------------')
        print('RDM task')
        print('time step: ' + str(self.dt))
        print('------------------------')

    def _new_trial(self, rng, dt):
        # TODO: why are rng and dt inputs?
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------

        fixation = 500
        measure = rng.choice([500, 580, 660, 760, 840, 920, 1000])
        gain = 1
        production = measure * gain
        ready = set = 83  # duration of ready and set cue
        tmax = self.fixation + measure + set + 2*production

        durations = {
            'fixation':  (0, self.fixation),
            'ready': (self.fixation, self.fixation + ready),
            'measure': (self.fixation, self.fixation + measure),
            'set': (self.fixation + measure,
                    self.fixation + measure + set),
            'production': (self.fixation + measure + set,
                           self.fixation + measure + set + 2*production),
            'tmax': tmax
            }

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        return {
            'durations': durations,
            'measure': measure,
            'production': production,
            }

    def step(self, action):
        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'continue': True}
        reward = 0
        tr_perf = False
        # TODO: why do we use t-1? Remove all
        # TODO: What's info continue? is it necessary? keep
        # TODO: why not call info info, like gym? change to info
        # TODO: why use integer t instead of actual t?
        # TODO: do we have intertrial interval in the beginning? can the network fixate at time 0?
        if not self.in_epoch(self.t, 'production'):
            if action != self.actions['FIXATE']:
                info['continue'] = False
                reward = self.R_ABORTED
        else:
            if action == self.actions['GO']:
                info['continue'] = False  # terminate
                # actual production time
                t_prod = self.t*self.dt - trial['durations']['measure'][1]
                eps = abs(t_prod - trial['production'])
                eps_threshold = 0.2*trial['production']+25
                if eps > eps_threshold:
                    info['correct'] = False
                    reward = self.R_FAIL
                else:
                    info['correct'] = True
                    reward = (1. - eps/eps_threshold)**1.5
                    reward = min(reward, 0.1)
                    reward *= self.R_CORRECT
                tr_perf = True

                info['t_choice'] = self.t

        # ---------------------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------------------

        obs = np.zeros(len(self.inputs))
        obs[self.inputs['FIXATION']] = 1  # TODO: this is always on now
        if self.in_epoch(self.t, 'ready'):
            obs[self.inputs['READY']] = 1
        if self.in_epoch(self.t, 'set'):
            obs[self.inputs['SET']] = 1

        # ---------------------------------------------------------------------
        # new trial?
        # TODO: it's confusing that the 4-th input is continue, but the function wants "info"
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
            self.trial = self._new_trial(self.rng, self.dt)
        else:
            self.t += 1

        done = False  # TODO: revisit
        return obs, reward, done, info

    def terminate(perf):
        p_decision, p_correct = tasktools.correct_2AFC(perf)

        return p_decision >= 0.99 and p_correct >= 0.8

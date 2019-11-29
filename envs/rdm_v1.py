#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:55:36 2019

@author: molano
"""

from neurogym.envs import ngym
from neurogym.ops import tasktools
import numpy as np
from gym import spaces


class RDM(ngym.ngym):
    def __init__(self, dt=100,
                 fixation_min=300,
                 fixation_max=700,
                 stimulus_min=80,
                 stimulus_mean=330,
                 stimulus_max=1500,
                 decision=500,
                 p_catch=0.1,
                 stimEv=1.,
                 **kwargs):
        super().__init__(dt=dt)
        # Actions (fixate, left, right)
        # self.actions = [0, -1, 1]
        # trial conditions (left, right)
        # self.choices = [-1, 1]
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)
        # TODO: this is not natural
        self.fixation_mean = (fixation_max + fixation_min)/2
        self.fixation_min = fixation_min
        self.fixation_max = fixation_max
        self.stimulus_min = stimulus_min
        self.stimulus_mean = stimulus_mean
        self.stimulus_max = stimulus_max
        self.decision = decision
        self.mean_trial_duration = self.fixation_mean + self.stimulus_mean +\
            self.decision
        # TODO: How to make this easier?
        self.max_trial_duration = self.fixation_max + self.stimulus_max +\
            self.decision
        self.max_steps = int(self.max_trial_duration/dt)
        self.p_catch = p_catch
        if (self.fixation_mean == 0 or self.decision == 0 or
           self.stimulus_mean == 0):
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXX')
        print('Random Dots Motion Task')
        print('Mean Fixation: ' + str(self.fixation_mean))
        print('Min Stimulus Duration: ' + str(self.stimulus_min))
        print('Mean Stimulus Duration: ' + str(self.stimulus_mean))
        print('Max Stimulus Duration: ' + str(self.stimulus_max))
        print('Decision: ' + str(self.decision))
        print('(time step: ' + str(self.dt) + ')')
        print('XXXXXXXXXXXXXXXXXXXXXX')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        # start new trial
        # self.trial = self._new_trial()
        self._new_trial()

    def _new_trial(self):
        """
        _new_trial() is called when a trial ends to get the specifications of
        the next trial. Such specifications are stored in a dictionary with
        the following items:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
        """
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                   self.stimulus_mean,
                                                   xmin=self.stimulus_min,
                                                   xmax=self.stimulus_max)
        fixation = self.rng.uniform(self.fixation_min, self.fixation_max)
        decision = self.decision

        # Introduce catch trials
        if self.rng.random() < self.p_catch:
            fixation = self.max_trial_duration
            stimulus = 0
            decision = 0
            self.catch = True
        else:
            self.catch = False
        # maximum length of current trial
        self.tmax = fixation + stimulus + decision
        durations = {
            'fixation': (0, fixation),
            'stimulus': (fixation, fixation + stimulus),
            'decision': (fixation + stimulus,
                         fixation + stimulus + decision),
            }
        self.fixation_0 = 0
        self.fixation_1 = fixation
        self.stimulus_0 = fixation
        self.stimulus_1 = fixation + stimulus
        self.decision_0 = fixation + stimulus
        self.decision_1 = fixation + stimulus + decision
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice(self.choices)
        coh = self.rng.choice(self.cohs)
        self.durations = durations
        self.ground_truth = ground_truth
        self.coh = coh
        t = np.arange(0, self.tmax, self.dt)

        obs = np.zeros((len(t), 3))

        fixation_period = np.logical_and(t >= self.fixation_0,
                                         t < self.fixation_1)
        stimulus_period = np.logical_and(t >= self.stimulus_0,
                                         t < self.stimulus_1)
        decision_period = np.logical_and(t >= self.decision_0,
                                         t < self.decision_1)

        obs[fixation_period, 0] = 1
        n_stim = int(stimulus/self.dt)
        obs[stimulus_period, 0] = 1
        obs[stimulus_period, ground_truth] = (1 + coh/100)/2
        obs[stimulus_period, 3 - ground_truth] = (1 - coh/100)/2
        obs[stimulus_period] += np.random.randn(n_stim, 3) * self.sigma_dt
        self.obs = obs
        self.t = 0
        self.num_tr += 1
        self.gt = np.zeros((len(t),), dtype=np.int)
        self.gt[decision_period] = self.ground_truth

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        # TODO: Try pre-generating stimulus for each trial
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = np.zeros((3,))
        if self.fixation_0 <= self.t < self.fixation_1:
            gt[0] = 1
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.decision_0 <= self.t < self.decision_1:
            gt[self.ground_truth] = 1
            if self.ground_truth == action:
                reward = self.R_CORRECT
            elif self.ground_truth == 3 - action:  # 3-action is the other act
                reward = self.R_FAIL
            new_trial = action != 0
        else:
            pass

        obs = self.obs[int(self.t/self.dt), :]

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax,
                                                self.dt,
                                                new_trial,
                                                self.R_MISS, reward)
        self.t += self.dt

        done = self.num_tr > self.num_tr_exp

        return obs, reward, done, {'new_trial': new_trial}

    def step(self, action):
        """
        step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        Note that the main computations are done by the function _step(action),
        and the extra lines are basically checking whether to call the
        _new_trial() function in order to start a new trial
        """
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self._new_trial()
        return obs, reward, done, info

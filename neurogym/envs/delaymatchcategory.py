#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to sample

"""
from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


class DelayedMatchCategory(ngym.EpochEnv):
    def __init__(self,
                 dt=100,
                 tmax=3500,
                 fixation=500,
                 sample=500,
                 delay=1500,
                 test=500,
                 decision=500,
                 ):
        super().__init__(dt=dt)
        self.choices = [1, 2]  # match, non-match

        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        # TODO: Find these info from a paper
        self.tmax = tmax
        self.fixation = fixation
        self.sample = sample
        self.delay = delay
        self.test = test
        self.decision = decision
        self.mean_trial_duration = self.tmax

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        # Fixation + Match + Non-match
        self.action_space = spaces.Discrete(3)
        # Fixation + cos(theta) + sin(theta)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

    def __str__(self):
        string = 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration / self.dt)
        return string

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        if 'gt' in kwargs.keys():
            ground_truth = kwargs['gt']
        else:
            ground_truth = self.rng.choice(self.choices)

        sample_category = self.rng.choice([0, 1])
        sample_theta = (sample_category + self.rng.random()) * np.pi

        test_category = sample_category
        if ground_truth == 2:
            test_category = 1 - test_category
        test_theta = (test_category + self.rng.random()) * np.pi

        self.ground_truth = ground_truth

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------


        if 'durs' in kwargs.keys():
            fixation = kwargs['durs'][0]
            sample = kwargs['durs'][1]
            delay = kwargs['durs'][2]
            test = kwargs['durs'][3]
            decision = kwargs['durs'][4]
        else:
            # stimulus = tasktools.trunc_exp(self.rng, self.dt,
            #                                self.stimulus_mean,
            #                                xmin=self.stimulus_min,
            #                                xmax=self.stimulus_max)
            fixation = self.fixation
            sample = self.sample
            delay = self.delay
            test = self.test
            decision = self.decision

        # maximum length of current trial
        self.tmax = fixation + sample + delay + test + decision

        self.add_epoch('fixation', start=0, duration=fixation)
        self.add_epoch('sample', start=fixation, duration=sample)
        self.add_epoch('delay', start=fixation+sample, duration=delay)
        self.add_epoch('test', start=fixation+sample+delay, duration=test)
        self.add_epoch('decision', start=fixation+sample+delay+test, duration=decision)

        # self.fixation_0 = 0
        # self.fixation_1 = fixation
        # self.sample_0 = fixation
        # self.sample_1 = fixation + sample
        # self.delay_0 = fixation + sample
        # self.delay_1 = fixation + sample + delay
        # self.test_0 = fixation + sample + delay
        # self.test_1 = fixation + sample + delay + test
        # self.decision_0 = fixation + sample + delay + test
        # self.decision_1 = fixation + sample + delay + test + decision

        t = np.arange(0, self.tmax, self.dt)
        obs = np.zeros((len(t), 3))

        fixation_period = np.logical_and(t >= self.fixation_0, t < self.fixation_1)
        sample_period = np.logical_and(t >= self.sample_0, t < self.sample_1)
        delay_period = np.logical_and(t >= self.delay_0, t < self.delay_1)
        test_period = np.logical_and(t >= self.test_0, t < self.test_1)
        decision_period = np.logical_and(t >= self.decision_0, t < self.decision_1)

        # self.add_ob('sample', value=[0, np.cos(sample_theta), np.sin(sample_theta)])

        obs[:, 0] = 1
        obs[decision_period, 0] = 0

        obs[sample_period, 1] = np.cos(sample_theta)
        obs[sample_period, 2] = np.sin(sample_theta)

        obs[test_period, 1] = np.cos(test_theta)
        obs[test_period, 2] = np.sin(test_theta)

        obs[:, 1:] += np.random.randn(len(t), 2) * self.sigma_dt

        self.obs = obs

        self.t = 0
        self.num_tr += 1

    def _step(self, action, **kwargs):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        if self.num_tr == 0:
            # start first trial
            self.new_trial()
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
            elif self.ground_truth != 0:  # 3-action is the other act
                reward = self.R_FAIL
            new_trial = action != 0
        else:
            gt[0] = 1
        obs = self.obs[int(self.t/self.dt), :]

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax,
                                                self.dt, new_trial,
                                                self.R_MISS, reward)
        self.t += self.dt

        done = self.num_tr > self.num_tr_exp

        return obs, reward, done, {'new_trial': new_trial, 'gt': gt}

    def step(self, action):
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.new_trial()
        return obs, reward, done, info


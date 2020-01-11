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

    def __str__(self):
        string = 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += 'max num. steps: ' + str(self.mean_trial_duration / self.dt)
        return string

    def _new_trial(self, **kwargs):
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
            fixation = self.fixation
            sample = self.sample
            delay = self.delay
            test = self.test
            decision = self.decision

        self.add_epoch('fixation', duration=fixation, start=0)
        self.add_epoch('sample', duration=sample, after='fixation')
        self.add_epoch('delay', duration=delay, after='sample')
        self.add_epoch('test', duration=test, after='delay')
        self.add_epoch('decision', duration=decision, after='test', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('sample', [1, np.cos(sample_theta), np.sin(sample_theta)])
        self.set_ob('delay', [1, 0, 0])
        self.set_ob('test', [1, np.cos(test_theta), np.sin(test_theta)])
        self.set_ob('decision', [0, 0, 0])

        self.obs[:, 1:] += np.random.randn(*self.obs[:, 1:].shape) * self.sigma_dt

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
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = np.zeros((3,))
        if self.in_epoch('fixation'):
            gt[0] = 1
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            gt[self.ground_truth] = 1
            if self.ground_truth == action:
                reward = self.R_CORRECT
            elif self.ground_truth != 0:  # 3-action is the other act
                reward = self.R_FAIL
            new_trial = action != 0
        else:
            gt[0] = 1
        obs = self.obs[self.t_ind, :]

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


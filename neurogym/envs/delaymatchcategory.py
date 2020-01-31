#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to category

"""
from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym


class DelayedMatchCategory(ngym.EpochEnv):
    metadata = {
        'description': 'A sample stimulus is followed by a delay and test.' +
        ' Agents are required to indicate if the sample and test are in the' +
        ' same category.',
        'paper_link': 'https://www.nature.com/articles/nature05078',
        'paper_name': '''Experience-dependent representation
        of visual categories in parietal cortex''',
        'timing': {
            'fixation': ('constant', 500),
            'sample': ('constant', 650),
            'first_delay': ('constant', 1000),
            'test': ('constant', 650)},
        # 'second_delay': ('constant', 250),  # TODO: not implemented
        # 'decision': ('constant', 650)},  # TODO: not implemented
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised setting']
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2]  # match, non-match

        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.abort = False

        # Fixation + Match + Non-match
        self.action_space = spaces.Discrete(3)

        # Fixation + cos(theta) + sin(theta)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_category': self.rng.choice([0, 1]),
        }
        self.trial.update(**kwargs)

        ground_truth = self.trial['ground_truth']
        sample_category = self.trial['sample_category']

        sample_theta = (sample_category + self.rng.random()) * np.pi

        test_category = sample_category
        if ground_truth == 2:
            test_category = 1 - test_category
        test_theta = (test_category + self.rng.random()) * np.pi

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('sample', after='fixation')
        self.add_epoch('first_delay', after='sample')
        self.add_epoch('test', after='first_delay', last_epoch=True)
        # self.add_epoch('decision', after='test', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])

        ob = self.view_ob('sample')
        ob[:, :] += np.array([1, np.cos(sample_theta), np.sin(sample_theta)])
        ob[:, 1:] += np.random.randn(ob.shape[0], 2) * self.sigma_dt

        self.set_ob('first_delay', [1, 0, 0])

        ob = self.view_ob('test')
        ob[:, :] += np.array([1, np.cos(test_theta), np.sin(test_theta)])
        ob[:, 1:] += np.random.randn(ob.shape[0], 2) * self.sigma_dt

        # self.set_ob('test', [0, 0, 0])

        self.set_groundtruth('test', ground_truth)

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

        obs = self.obs_now
        gt = self.gt_now

        reward = 0
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('test'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}

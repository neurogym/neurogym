#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym


class DelayMatchCategory(ngym.PeriodEnv):
    r"""Delay match-to-category task.

    A sample stimulus is followed by a delay and test. Agents are required
    to indicate if the sample and test are in the same category.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature05078',
        'paper_name': '''Experience-dependent representation
        of visual categories in parietal cortex''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=2):
        super().__init__(dt=dt)
        self.choices = ['match', 'non-match']  # match, non-match

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),
            'sample': ('constant', 650),
            'first_delay': ('constant', 1000),
            'test': ('constant', 650)}
        # 'second_delay': ('constant', 250),  # TODO: not implemented
        # 'decision': ('constant', 650)},  # TODO: not implemented}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + dim_ring,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'match': 1, 'non-match': 2}

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_category': self.rng.choice([0, 1]),
        }
        self.trial.update(**kwargs)

        ground_truth = self.trial['ground_truth']
        sample_category = self.trial['sample_category']
        if ground_truth == 'match':
            test_category = sample_category
        else:
            test_category = 1 - sample_category

        sample_theta = (sample_category + self.rng.rand()) * np.pi
        test_theta = (test_category + self.rng.rand()) * np.pi

        stim_sample = np.cos(self.theta - sample_theta) * 0.5 + 0.5
        stim_test = np.cos(self.theta - test_theta) * 0.5 + 0.5

        # Periods
        periods = ['fixation', 'sample', 'first_delay', 'test']
        self.add_period(periods, after=0, last_period=True)
        # self.add_period('decision', after='test', last_period=True)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'test', where='fixation')
        self.add_ob(stim_sample, 'sample', where='stimulus')
        self.add_ob(stim_test, 'test', where='stimulus')
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus')

        self.set_groundtruth(self.act_dict[ground_truth], 'test')

    def _step(self, action, **kwargs):
        new_trial = False

        obs = self.ob_now
        gt = self.gt_now

        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('test'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = DelayMatchCategory()
    ngym.utils.plot_env(env, num_steps=100)
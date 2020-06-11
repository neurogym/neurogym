#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class DelayMatchSample(ngym.PeriodEnv):
    r"""Delay-match-to-sample.

    A sample stimulus is followed by a delay and test. Agents are required
    to indicate if the sample and test are the same stimulus.
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/' +
                      '5154.full.pdf',
        'paper_name': '''Neural Mechanisms of Visual Working Memory
    in Prefrontal Cortex of the Macaque''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=2):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 300),
            'sample': ('constant', 500),
            'delay': ('constant', 1000),
            'test': ('constant', 500),
            'decision': ('constant', 900)}
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
        # Trial
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_theta': self.rng.choice(self.theta),
        }
        self.trial.update(kwargs)

        ground_truth = self.trial['ground_truth']
        sample_theta = self.trial['sample_theta']
        if ground_truth == 1:
            test_theta = sample_theta
        else:
            test_theta = np.mod(sample_theta + np.pi, 2 * np.pi)
        self.trial['test_theta'] = test_theta

        stim_sample = np.cos(self.theta - sample_theta) * 0.5 + 0.5
        stim_test = np.cos(self.theta - test_theta) * 0.5 + 0.5

        # Periods
        self.add_period(['fixation', 'sample', 'delay', 'test', 'decision'],
                        after=0, last_period=True)

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        self.add_ob(stim_sample, 'sample', where='stimulus')
        self.add_ob(stim_test, 'test', where='stimulus')
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus')

        self.set_groundtruth(ground_truth, 'decision')

    def _step(self, action):
        new_trial = False
        reward = 0

        obs = self.ob_now
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class DelayMatchSampleDistractor1D(ngym.PeriodEnv):
    r"""Delay Match to sample with multiple, potentially repeating distractors.

    Args:
        dt: Timestep duration. (def: 100 (ms), int)
        rewards: dictionary of rewards
        timing: Description and duration of periods forming a trial.
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/' +
        '5154.full.pdf',
        'paper_name': '''Neural Mechanisms of Visual Working Memory
        in Prefrontal Cortex of the Macaque''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)
        self.choices = [1, 2, 3]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 300),
            'sample': ('constant', 500),
            'delay1': ('constant', 1000),
            'test1': ('constant', 500),
            'delay2': ('constant', 1000),
            'test2': ('constant', 500),
            'delay3': ('constant', 1000),
            'test3': ('constant', 500)}
        if timing:
            self.timing.update(timing)

        self.abort = False
        self.action_space = spaces.Discrete(2)
        self.act_dict = {'fixation': 0, 'match': 1}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, 33)}
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            # There is always a match, ground_truth is which test is a match
            'ground_truth': self.rng.choice(self.choices),
            'sample': self.rng.uniform(0, 2*np.pi),
        }
        self.trial.update(kwargs)

        ground_truth = self.trial['ground_truth']
        sample = self.trial['sample']
        for i in [1, 2, 3]:
            tmp = sample if i == ground_truth else self.rng.uniform(0, 2*np.pi)
            self.trial['test'+str(i)] = tmp

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'sample', 'delay1', 'test1',
                   'delay2', 'test2', 'delay3', 'test3']
        self.add_period(periods, after=0, last_period=True)

        self.add_ob(1, 'fixation', where='fixation')
        for period in ['sample', 'test1', 'test2', 'test3']:
            self.add_ob(np.cos(self.theta - self.trial[period]), period, 'stimulus')

        self.set_groundtruth(1, 'test'+str(ground_truth))

    def _step(self, action):
        new_trial = False
        reward = 0

        obs = self.ob_now
        gt = self.gt_now
        if ((self.in_period('fixation') or self.in_period('sample'))
           and action != 0):
            reward = self.rewards['abort']
            new_trial = self.abort
        elif not self.in_period('test'+str(self.trial['ground_truth'])):
            if action != 0:
                reward = self.rewards['fail']
                new_trial = True
        else:
            if action == 1:
                reward = self.rewards['correct']
                new_trial = True
                self.performance = 1

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}

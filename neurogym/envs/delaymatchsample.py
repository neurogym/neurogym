#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


class DelayMatchSample(ngym.TrialEnv):
    r"""Delayed match-to-sample task.

    A sample stimulus is shown during the sample period. The stimulus is
    characterized by a one-dimensional variable, such as its orientation
    between 0 and 360 degree. After a delay period, a test stimulus is
    shown. The agent needs to determine whether the sample and the test
    stimuli are equal, and report that decision during the decision period.
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/' +
                      '5154.full.pdf',
        'paper_name': '''Neural Mechanisms of Visual Working Memory in 
        Prefrontal Cortex of the Macaque''',
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
            'fixation': 300,
            'sample': 500,
            'delay': 1000,
            'test': 500,
            'decision': 900}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]

        name = {'fixation': 0, 'stimulus': range(1, dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + dim_ring,), dtype=np.float32, name=name)

        name = {'fixation': 0, 'match': 1, 'non-match': 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        # Trial
        trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample_theta': self.rng.choice(self.theta),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        sample_theta = trial['sample_theta']
        if ground_truth == 1:
            test_theta = sample_theta
        else:
            test_theta = np.mod(sample_theta + np.pi, 2 * np.pi)
        trial['test_theta'] = test_theta

        stim_sample = np.cos(self.theta - sample_theta) * 0.5 + 0.5
        stim_test = np.cos(self.theta - test_theta) * 0.5 + 0.5

        # Periods
        self.add_period(['fixation', 'sample', 'delay', 'test', 'decision'])

        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')
        self.add_ob(stim_sample, 'sample', where='stimulus')
        self.add_ob(stim_test, 'test', where='stimulus')
        self.add_randn(0, self.sigma, ['sample', 'test'], where='stimulus')

        self.set_groundtruth(ground_truth, 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0

        ob = self.ob_now
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

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


class DelayMatchSampleDistractor1D(ngym.TrialEnv):
    r"""Delayed match-to-sample with multiple, potentially repeating
    distractors.

    A sample stimulus is shown during the sample period. The stimulus is
    characterized by a one-dimensional variable, such as its orientation
    between 0 and 360 degree. After a delay period, the first test stimulus is
    shown. The agent needs to determine whether the sample and this test
    stimuli are equal. If so, it needs to produce the match response. If the
    first test is not equal to the sample stimulus, another delay period and
    then a second test stimulus follow, and so on.
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
            'fixation': 300,
            'sample': 500,
            'delay1': 1000,
            'test1': 500,
            'delay2': 1000,
            'test2': 500,
            'delay3': 1000,
            'test3': 500}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        name = {'fixation': 0, 'stimulus': range(1, 33)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,),
                                            dtype=np.float32, name=name)

        name = {'fixation': 0, 'match': 1}
        self.action_space = spaces.Discrete(2, name=name)

    def _new_trial(self, **kwargs):
        trial = {
            # There is always a match, ground_truth is which test is a match
            'ground_truth': self.rng.choice(self.choices),
            'sample': self.rng.uniform(0, 2*np.pi),
        }
        trial.update(kwargs)

        ground_truth = trial['ground_truth']
        sample = trial['sample']
        for i in [1, 2, 3]:
            tmp = sample if i == ground_truth else self.rng.uniform(0, 2*np.pi)
            trial['test'+str(i)] = tmp

        periods = ['fixation', 'sample', 'delay1', 'test1',
                   'delay2', 'test2', 'delay3', 'test3']
        self.add_period(periods)

        self.add_ob(1, 'fixation', where='fixation')
        for period in ['sample', 'test1', 'test2', 'test3']:
            self.add_ob(np.cos(self.theta - trial[period]), period, 'stimulus')

        self.set_groundtruth(1, 'test'+str(ground_truth))

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0

        ob = self.ob_now
        gt = self.gt_now
        if ((self.in_period('fixation') or self.in_period('sample')) and
                action != 0):
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

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}

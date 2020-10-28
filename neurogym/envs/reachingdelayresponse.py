#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import neurogym as ngym
from neurogym import spaces


# TODO: Task need to be revisited
class ReachingDelayResponse(ngym.TrialEnv):
    r"""Reaching task with a delay period.

    A reaching direction is presented by the stimulus during the stimulus
    period. Followed by a delay period, the agent needs to respond to the
    direction of the stimulus during the decision period.
    """
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'tags': ['perceptual', 'delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None,
                 lowbound=0., highbound=1.):
        super().__init__(dt=dt)
        self.lowbound = lowbound
        self.highbound = highbound

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.,
                        'fail': -0., 'miss': -0.5}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'stimulus': 500,
            'delay': (0, 1000, 2000),
            'decision': 500}
        if timing:
            self.timing.update(timing)

        self.r_tmax = self.rewards['miss']
        self.abort = False

        name = {'go': 0, 'stimulus': 1}
        self.observation_space = spaces.Box(low=np.array([0., -2]),
                                            high=np.array([1, 2.]),
                                            dtype=np.float32, name=name)

        self.action_space = spaces.Box(low=np.array((-1.0, -1.0)),
                                       high=np.array((1.0, 2.0)),
                                       dtype=np.float32)

    def _new_trial(self, **kwargs):
        # Trial
        trial = {
            'ground_truth': self.rng.uniform(self.lowbound, self.highbound)
        }
        trial.update(kwargs)
        ground_truth_stim = trial['ground_truth']

        # Periods
        self.add_period(['stimulus', 'delay', 'decision'])

        self.add_ob(ground_truth_stim, 'stimulus', where='stimulus')
        self.set_ob([0, -0.5], 'delay')
        self.set_ob([1, -0.5], 'decision')

        self.set_groundtruth([-1, -0.5], ['stimulus', 'delay'])
        self.set_groundtruth([1, ground_truth_stim], 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now  # 2 dim now

        if self.in_period('stimulus'):
            if not action[0] < 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action[0] > 0:
                new_trial = True
                reward = self.rewards['correct']/((1+abs(action[1]-gt[1]))**2)
                self.performance = reward/self.rewards['correct']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

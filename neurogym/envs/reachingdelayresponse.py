#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from gym import spaces

import neurogym as ngym


class ReachingDelayResponse(ngym.PeriodEnv):
    metadata = {
        'description': 'Working memory visual spatial task ' +
        ' ~ Funahashi et al. 1991 adapted to freely moving mice in a ' +
        'continous choice-space.\n' +
        'Brief description: while fixating, stimulus is presented in a ' +
        'touchscreen (bright circle). Afterwards (perhaps including an ' +
        'extra delay), doors open allowing the mouse to touch the screen ' +
        'where the stimulus was located.',
        'paper_link': None,
        'paper_name': None,
        'tags': ['perceptual', 'delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, lowbound=0.,
                 highbound=1.):
        """
        Working memory visual spatial task ~ Funahashi et al. 1991 adapted to
        freely moving mice in a continous choice-space.
        """
        super().__init__(dt=dt)
        self.lowbound = lowbound
        self.highbound = highbound
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -0., 'miss': -0.5}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'stimulus': ('constant', 500),
            'delay': ('choice', [0, 1000, 2000]),
            'decision': ('constant', 5000)}
        if timing:
            self.timing.update(timing)

        self.r_tmax = self.rewards['miss']
        self.abort = False

        self.action_space = spaces.Box(low=np.array((-1.0, -1.0)),
                                       high=np.array((1.0, 2.0)),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0., -2]),
                                            high=np.array([1, 2.]),
                                            dtype=np.float32)
        self.ob_dict = {'go': 0, 'stimulus': 1}

    def new_trial(self, **kwargs):
        # Trial
        self.trial = {
            'ground_truth': self.rng.uniform(self.lowbound, self.highbound)
        }
        self.trial.update(kwargs)
        ground_truth_stim = self.trial['ground_truth']

        # Periods
        self.add_period(['stimulus', 'delay', 'decision'], after=0, last_period=True)

        self.add_ob(ground_truth_stim, 'stimulus', where='stimulus')
        self.set_ob([0, -0.5], 'delay')
        self.set_ob([1, -0.5], 'decision')

        self.set_groundtruth([-1, -0.5], ['stimulus', 'delay'])
        self.set_groundtruth([1, ground_truth_stim], 'decision')

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

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ReachingDelayResponse()
    ngym.utils.plot_env(env, num_steps_env=100)

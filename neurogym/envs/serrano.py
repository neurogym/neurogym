#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from gym import spaces
from neurogym.meta import tasks_info
import neurogym as ngym


class Serrano(ngym.EpochEnv):
    metadata = {
        'description': '',
        'paper_link': None,
        'paper_name': None,
        'timing': {
            'stimulus': ('constant', 100),
            'delay': ('choice', [0, 100, 200]),
            'decision': ('constant', 300)},
        'tags': ['perceptual', 'delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised setting']
    }

    def __init__(self, dt=100, timing=None, **kwargs):
        super().__init__(dt=dt, timing=timing)
        self.lowbound = 0.
        self.highbound = 1.0
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        # Reards
        self.R_ABORTED = -0.1
        self.R_CORRECT = 1.
        self.R_FAIL = 0.
        self.r_tmax = -0.5
        self.abort = False

        self.action_space = spaces.Box(low=np.array((-1.0, -1.0)),
                                       high=np.array((1.0, 2.0)),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0., -2]),
                                            high=np.array([1, 2.]),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # Trial
        self.trial = {
            'ground_truth': np.random.uniform(self.lowbound, self.highbound)
        }
        self.trial.update(kwargs)
        ground_truth = self.trial['ground_truth']

        # Epochs
        self.add_epoch('stimulus', after=0)
        self.add_epoch('delay', after='stimulus')
        self.add_epoch('decision', after='delay', last_epoch=True)

        stimulus = self.view_ob('stimulus')
        stimulus[:, 1] = ground_truth
        stimulus[:, 1] +=\
            np.random.rand(stimulus.shape[0])*self.sigma_dt*0.01

        self.set_ob('delay', [0, -0.5])
        self.set_ob('decision', [1, -0.5])
        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        """ not a dictionary anymore"""
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        # observations
        """elif self.in_epoch('delay'):
            if (action[1] < -1) or (action[1] > 2):
                reward = self.R_ABORTED*0.2"""

        if self.in_epoch('stimulus'):
            if not action[0] < 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action[0] > 0:
                new_trial = True
                reward = self.R_CORRECT/((1+abs(action[1]-gt))**2)
                assert reward <= 1, f'{action}, {gt} {reward}'

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = Serrano()
    tasks_info.plot_struct(env, num_steps_env=20, n_stps_plt=20)

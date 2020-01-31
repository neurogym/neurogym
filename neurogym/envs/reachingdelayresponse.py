#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from gym import spaces
from neurogym.meta import info
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
        'timing': {
            'stimulus': ('constant', 500),
            'delay': ('choice', [0, 1000, 2000]),
            'decision': ('constant', 5000)},
        'tags': ['perceptual', 'delayed response', 'continuous action space',
                 'multidimensional action space', 'supervised']
    }

    def __init__(self, dt=100, timing=None, lowbound=0., highbound=1.):
        super().__init__(dt=dt, timing=timing)
        self.lowbound = lowbound
        self.highbound = highbound
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
        ground_truth_stim = self.trial['ground_truth']

        # Periods
        self.add_period('stimulus', after=0)
        self.add_period('delay', after='stimulus')
        self.add_period('decision', after='delay', last_period=True)

        stimulus = self.view_ob('stimulus')
        stimulus[:, 1] = ground_truth_stim
        stimulus[:, 1] +=\
            np.random.rand(stimulus.shape[0])*self.sigma_dt

        gt = self.view_groundtruth('stimulus')
        for ep in ['stimulus', 'delay']:
            gt[:,0] = -1. # fixate
            gt[:,1] = -0.5 # no stim ~ arbitrary number which can cause issues with regression

        self.set_ob('delay', [0, -0.5])
        self.set_ob('decision', [1, -0.5])
        decision_gt = self.view_groundtruth('decision')
        decision_gt[:,0] = 1. # go
        decision_gt[:,1] = ground_truth_stim # Where to respond

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now # 2 dim now

        if self.in_period('stimulus'):
            if not action[0] < 0:
                new_trial = self.abort
                reward = self.R_ABORTED        
        elif self.in_period('decision'):
            if action[0] > 0:
                new_trial = True
                reward = self.R_CORRECT/((1+abs(action[1]-gt[1]))**2)

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = ReachingDelayResponse()
    info.plot_struct(env, num_steps_env=20, n_stps_plt=20)
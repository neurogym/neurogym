#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delay Match to sample

"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym


class DelayedMatchToSample(ngym.EpochEnv):
    metadata = {
        'description': 'A sample stimulus is followed by a delay and test.' +
        ' Agents are required to indicate if the sample and test are the' +
        ' same stimulus.',
        'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/' +
        '5154.full.pdf',
        'paper_name': '''Neural Mechanisms of Visual Working Memory
        in Prefrontal Cortex of the Macaque''',
        'timing': {
            'fixation': ('constant', 300),
            'sample': ('constant', 500),
            'delay': ('constant', 1000),
            'test': ('constant', 500),
            'decision': ('constant', 900)},
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        # TODO: Code a continuous space version
        self.choices = [1, 2]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = self.sigma/np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'sample': self.rng.choice([1, 2]),
        }
        self.trial.update(kwargs)

        ground_truth = self.trial['ground_truth']
        sample = self.trial['sample']

        test = sample if ground_truth == 1 else 3 - sample
        self.trial['test'] = test
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('sample', after='fixation')
        self.add_epoch('delay', after='sample')
        self.add_epoch('test', after='delay')
        self.add_epoch('decision', after='test', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        ob = self.view_ob('sample')
        ob[:, 0] = 1
        ob[:, sample] = 1
        ob[:, 1:] += np.random.randn(ob.shape[0], 2) * self.sigma_dt

        ob = self.view_ob('test')
        ob[:, 0] = 1
        ob[:, test] = 1
        ob[:, 1:] += np.random.randn(ob.shape[0], 2) * self.sigma_dt

        self.set_ob('delay', [1, 0, 0])

        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        new_trial = False
        reward = 0

        obs = self.obs_now
        gt = self.gt_now

        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


class DelayedMatchToSampleDistractor1D(ngym.EpochEnv):
    """
    Delay Match to sample with multiple, potentially repeating distractors
    """
    metadata = {
        'description': '''Delay Match to sample with multiple,
         potentially repeating distractors.''',
        'paper_link': 'https://www.jneurosci.org/content/jneuro/16/16/' +
        '5154.full.pdf',
        'paper_name': '''Neural Mechanisms of Visual Working Memory
        in Prefrontal Cortex of the Macaque''',
        'timing': {
            'fixation': ('constant', 300),
            'sample': ('constant', 500),
            'delay1': ('constant', 1000),
            'test1': ('constant', 500),
            'delay2': ('constant', 1000),
            'test2': ('constant', 500),
            'delay3': ('constant', 1000),
            'test3': ('constant', 500)},
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)
        self.choices = [1, 2, 3]
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = self.sigma/np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,),
                                            dtype=np.float32)
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
        # Epochs
        # ---------------------------------------------------------------------
        epochs = ['fixation', 'sample', 'delay1', 'test1',
                  'delay2', 'test2', 'delay3', 'test3']
        self.add_epoch(epochs[0], after=0)
        for i in range(1, len(epochs)):
            self.add_epoch(epochs[i], after=epochs[i-1],
                           last_epoch=i == len(epochs)-1)

        ob = self.view_ob('fixation')
        ob[:, 0] = 1

        for epoch in ['sample', 'test1', 'test2', 'test3']:
            ob = self.view_ob(epoch)
            ob[:, 1:] += np.cos(self.theta - self.trial[epoch])

        self.set_groundtruth('test'+str(ground_truth), 1)

    def _step(self, action):
        new_trial = False
        reward = 0

        obs = self.obs_now
        gt = self.gt_now

        if not self.in_epoch('test'+str(self.trial['ground_truth'])):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        else:
            if action == 1:
                reward = self.R_CORRECT
                new_trial = True

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    from neurogym.tests import test_run
    from neurogym.meta.tasks_info import plot_struct
    env = DelayedMatchToSampleDistractor1D()
    test_run(env)
    plot_struct(env)

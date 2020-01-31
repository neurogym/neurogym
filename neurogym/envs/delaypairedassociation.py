#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delayed paired association

TODO: Add paper
"""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.meta import tasks_info


class DelayPairedAssociation(ngym.EpochEnv):
    metadata = {
        'description': 'A sample is followed by a delay and a test.' +
        ' Agents have to report if the pair sample-test is a rewarded pair' +
        ' or not.',
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'timing': {
            'fixation': ('constant', 0),
            'stim1': ('constant', 1000),
            'delay_btw_stim': ('constant', 13000),
            'stim2': ('constant', 1000),
            'delay_aft_stim': ('constant', 1000),
            'decision': ('constant', 500)},
        'noise': '''Standard deviation of the Gaussian noise added to
        the stimulus. (def: 0.01)''',
        'tags': ['perceptual', 'working memory', 'go/no-go',
                 'supervised']
    }

    def __init__(self, dt=100, timing=None, noise=0.01):
        super().__init__(dt=dt, timing=timing)
        self.choices = [0, 1]
        # trial conditions
        self.dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        self.association = 0  # GO if np.diff(self.pair)[0]%2==self.association
        # Input noise
        sigma = np.sqrt(2*100*noise)
        self.sigma_dt = sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.abort = False
        # action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,),
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
        pair = self.rng.choice(self.dpa_pairs)
        print(pair)
        self.trial = {
            'pair': pair,
            'ground_truth': int(np.diff(pair)[0] % 2 == self.association),
        }
        self.trial.update(kwargs)
        pair = self.trial['pair']
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stim1', after='fixation')
        self.add_epoch('delay_btw_stim', after='stim1')
        self.add_epoch('stim2', after='delay_btw_stim')
        self.add_epoch('delay_aft_stim', after='stim2')
        self.add_epoch('decision', after='delay_aft_stim', last_epoch=True)
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.set_ob('fixation', [1, 0, 0, 0, 0])

        ob = self.view_ob('stim1')
        ob[:, 0] = 1
        ob[:, pair[0]] = 1 + np.random.randn(ob.shape[0]) * self.sigma_dt

        ob = self.view_ob('stim2')
        ob[:, 0] = 1
        ob[:, pair[1]] = 1 + np.random.randn(ob.shape[0]) * self.sigma_dt

        self.set_ob('delay_btw_stim', [1, 0, 0, 0, 0])
        self.set_ob('delay_aft_stim', [1, 0, 0, 0, 0])

        self.set_groundtruth('decision', self.trial['ground_truth'])

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
        obs = self.obs_now
        gt = self.gt_now
        # observations
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL
                new_trial = True

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = DelayPairedAssociation()
    tasks_info.plot_struct(env, num_steps_env=1000, n_stps_plt=1000)

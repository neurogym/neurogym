#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delayed paired association

TODO: Add paper
"""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


class DPA(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': '''Active information maintenance in working memory by a sensory cortex''',
        'default_timing': {
            'fixation': ('truncated_exponential', [500, 200, 800]),
            'stim1': ('truncated_exponential', [500, 200, 800]),
            'delay_btw_stim': ('truncated_exponential', [500, 200, 800]),
            'stim2': ('truncated_exponential', [500, 200, 800]),
            'delay_aft_stim': ('truncated_exponential', [500, 200, 800]),
            'decision': ('truncated_exponential', [500, 200, 800])},
    }

    def __init__(self, dt=100, timing=None, noise=0.01,
                 simultaneous_stim=False, **kwargs):
        super().__init__(dt=dt, timing=timing)
        self.choices = [0, 1]
        # trial conditions
        self.dpa_pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        self.association = 0  # GO if np.diff(self.pair)[0]%2==self.association
        # Input noise
        self.sigma = np.sqrt(2*100*noise)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)
        self.sim_stim = simultaneous_stim

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
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
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """

        if 'pair' in kwargs.keys():
            self.pair = kwargs['pair']
        else:
            self.pair = self.rng.choice(self.dpa_pairs)

        if np.diff(self.pair)[0] % 2 == self.association:
            ground_truth = 1
        else:
            ground_truth = 0

        self.ground_truth = ground_truth

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
        tmp = np.array([1, 0, 0, 0, 0])
        tmp[self.pair[0]] = 1
        self.set_ob('stim1', tmp)
        tmp = np.array([1, 0, 0, 0, 0])
        tmp[self.pair[1]] = 1
        self.set_ob('stim2', tmp)
        self.set_ob('delay_btw_stim', [1, 0, 0, 0, 0])
        self.set_ob('delay_aft_stim', [1, 0, 0, 0, 0])

        # TODO: Not happy about having to do this ugly thing
        self.obs[self.stim1_ind0:self.stim1_ind1, self.pair[0]] += np.random.randn(
            self.stim1_ind1 - self.stim1_ind0) * self.sigma_dt
        self.obs[self.stim2_ind0:self.stim2_ind1, self.pair[1]] += np.random.randn(
            self.stim2_ind1 - self.stim2_ind0) * self.sigma_dt

        self.set_groundtruth('decision', ground_truth)

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
    env = DPA()
    tasktools.plot_struct(env)

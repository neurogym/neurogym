#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delayed paired association

TODO: Add paper
"""

import numpy as np
from gym import spaces

import neurogym as ngym


class DelayPairedAssociation(ngym.PeriodEnv):
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
        'tags': ['perceptual', 'working memory', 'go-no-go',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, noise=0.01):
        """
        A sample is followed by a delay and a test. Agents have to report if
        the pair sample-test is a rewarded pair or not.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: -1., float)
            R_MISS:  given when not responding when a response was expected.
            (def: 0., float)
        timing: Description and duration of periods forming a trial.
        noise: Standard deviation of the Gaussian noise added to
        the stimulus. (def: 0.01, float)
        """
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
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1., 'miss': 0.}
        if rewards:
            self.rewards.update(rewards)

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
            the case of perceptualDecisionMaking: fixation, stimulus
            and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        pair = self.dpa_pairs[self.rng.choice(len(self.dpa_pairs))]
        self.trial = {
            'pair': pair,
            'ground_truth': int(np.diff(pair)[0] % 2 == self.association),
        }
        self.trial.update(kwargs)
        pair = self.trial['pair']
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stim1', 'delay_btw_stim', 'stim2',
                   'delay_aft_stim', 'decision']
        self.add_period(periods, after=0, last_period=True)
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # set observations
        self.set_ob([1, 0, 0, 0, 0], 'fixation')

        ob = self.view_ob('stim1')
        ob[:, 0] = 1
        ob[:, pair[0]] = 1 + self.rng.randn(ob.shape[0]) * self.sigma_dt

        ob = self.view_ob('stim2')
        ob[:, 0] = 1
        ob[:, pair[1]] = 1 + self.rng.randn(ob.shape[0]) * self.sigma_dt

        self.set_ob([1, 0, 0, 0, 0], 'delay_btw_stim')
        self.set_ob([1, 0, 0, 0, 0], 'delay_aft_stim')
        # set ground truth
        self.set_groundtruth(self.trial['ground_truth'], 'decision')

        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards['miss']*self.trial['ground_truth']
        self.performance = 1-self.trial['ground_truth']

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
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
                    self.performance = 0
                new_trial = True

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = DelayPairedAssociation()
    ngym.utils.plot_env(env, num_steps_env=1000, def_act=0)

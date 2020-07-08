#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gym import spaces

import neurogym as ngym


class DelayPairedAssociation(ngym.TrialEnv):
    r"""A sample is followed by a delay and a test. Agents have to report if
    the pair sample-test is a rewarded pair or not.
    """

    metadata = {
        'paper_link': 'https://elifesciences.org/articles/43191',
        'paper_name': 'Active information maintenance in working memory' +
        ' by a sensory cortex',
        'tags': ['perceptual', 'working memory', 'go-no-go',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)
        self.choices = [0, 1]
        # trial conditions
        self.pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        self.association = 0  # GO if np.diff(self.pair)[0]%2==self.association
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        # Durations (stimulus duration will be drawn from an exponential)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1., 'miss': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 0,
            'stim1': 1000,
            'delay_btw_stim': 1000,
            'stim2': 1000,
            'delay_aft_stim': 1000,
            'decision': 500}
        if timing:
            self.timing.update(timing)

        self.abort = False
        # action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.act_dict = {'fixation': 0, 'go': 1}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, 5)}

    def _new_trial(self, **kwargs):
        pair = self.pairs[self.rng.choice(len(self.pairs))]
        trial = {
            'pair': pair,
            'ground_truth': int(np.diff(pair)[0] % 2 == self.association),
        }
        trial.update(kwargs)
        pair = trial['pair']
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stim1', 'delay_btw_stim', 'stim2',
                   'delay_aft_stim', 'decision']
        self.add_period(periods)
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # set observations
        self.add_ob(1, where='fixation')
        self.add_ob(1, 'stim1', where=pair[0])
        self.add_ob(1, 'stim2', where=pair[1])
        self.set_ob(0, 'decision')
        # set ground truth
        self.set_groundtruth(trial['ground_truth'], 'decision')

        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards['miss']*trial['ground_truth']
        self.performance = 1-trial['ground_truth']

        return trial

    def _step(self, action, **kwargs):
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        obs = self.ob_now
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
    ngym.utils.plot_env(env, num_steps=1000, def_act=0)

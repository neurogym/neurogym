#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym


class PostDecisionWager(ngym.PeriodEnv):
    r"""Post-decision wagering task assessing confidence.

    Agents do a discrimination task (see PerceptualDecisionMaking). On a
    random half of the trials, the agent is given the option to abort
    the direction discrimination and to choose instead a small but
    certain reward associated with a action.

    Args:
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://science.sciencemag.org/content/324/5928/' +
        '759.long',
        'paper_name': '''Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex''',
        'tags': ['perceptual', 'delayed response', 'confidence']
    }

    def __init__(self, dt=100, rewards=None, timing=None, dim_ring=2, sigma=1.0):
        super().__init__(dt=dt)

        self.wagers = [True, False]
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)
        self.cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        self.rewards['sure'] = 0.7 * self.rewards['correct']

        self.timing = {
            'fixation': ('constant', 100),  # XXX: not specified
            # 'target':  ('constant', 0), # XXX: not implemented, not specified
            'stimulus': ('truncated_exponential', [180, 100, 900]),
            'delay': ('truncated_exponential', [1350, 1200, 1800]),
            'pre_sure': ('uniform', [500, 750]),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        self.action_space = spaces.Discrete(4)
        self.act_dict = {'fixation': 0, 'choice': [1, 2], 'sure': 3}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': [1, 2], 'sure': 3}

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'wager': self.rng.choice(self.wagers),
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)
        coh = self.trial['coh']
        ground_truth = self.trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        periods = ['fixation', 'stimulus', 'delay']
        self.add_period(periods, after=0)
        if self.trial['wager']:
            self.add_period('pre_sure', after='stimulus')
            self.add_period('sure', duration=10000, after='pre_sure')
        self.add_period('decision', after='delay', last_period=True)

        # Observations
        self.add_ob(1, ['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh / 200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus')
        if self.trial['wager']:
            self.add_ob(1, 'sure', where='sure')

        # Ground truth
        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        new_trial = False

        reward = 0
        gt = self.gt_now

        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            new_trial = True
            if action == 0:
                new_trial = False
            elif action == self.act_dict['sure']:
                if trial['wager']:
                    reward = self.rewards['sure']
                    norm_rew = ((reward-self.rewards['fail'])/
                                (self.rewards['correct']-self.rewards['fail']))
                    self.performance = norm_rew
                else:
                    reward = self.rewards['abort']
            else:
                if action == trial['ground_truth']:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = PostDecisionWager()
    env.seed(seed=0)
    ngym.utils.plot_env(env, num_steps=100)  # , def_act=0)

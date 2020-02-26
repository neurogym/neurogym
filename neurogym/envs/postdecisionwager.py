#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:31:14 2019

@author: molano

Perceptual decision-making with postdecision wagering, based on

  Representation of confidence associated with a decision by
  neurons in the parietal cortex.
  R. Kiani & M. N. Shadlen, Science 2009.

  http://dx.doi.org/10.1126/science.1169405

"""
from __future__ import division

import numpy as np
from gym import spaces

import neurogym as ngym


class PostDecisionWager(ngym.PeriodEnv):
    metadata = {
        'description': """Agents do a discrimination task(see
         PerceptualDecisionMaking). On a random half of the trials,
         the agent is given the option to abort the direction discrimination
         and to choose instead a small but certain reward associated with
         a action.""",
        'paper_link': 'https://science.sciencemag.org/content/324/5928/' +
        '759.long',
        'paper_name': '''Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex''',
        'tags': ['perceptual', 'delayed response', 'confidence']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        Agents do a discrimination task(see PerceptualDecisionMaking). On a
        random half of the trials, the agent is given the option to abort
        the direction discrimination and to choose instead a small but
        certain reward associated with a action.
        """
        super().__init__(dt=dt)
#        # Actions
#        self.actions = tasktools.to_map('FIXATE', 'CHOOSE-LEFT',
#                                        'CHOOSE-RIGHT', 'CHOOSE-SURE')
        # Actions (fixate, left, right, sure)
        self.actions = [0, -1, 1, 2]
        # trial conditions
        self.wagers = [True, False]
        self.choices = [-1, 1]
        self.cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]

        # Input noise
        sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

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
        self.rewards['sure'] = 0.7*self.rewards['correct']

        # set action and observation space
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4, ),
                                            dtype=np.float32)

    # Input scaling
    def scale(self, coh):
        return (1 + coh/100)/2

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Wager or no wager?
        # ---------------------------------------------------------------------
        self.trial = {
            'wager': self.rng.choice(self.wagers),
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods, after=0, last_period=True)
        if self.trial['wager']:
            self.add_period('pre_sure', after='stimulus')
            self.add_period('sure', duration=10000, after='pre_sure')

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        info = {'new_trial': False}
        # ground truth signal is not well defined in this task
        info['gt'] = np.zeros((4,))
        # rewards
        reward = 0
        # observations
        obs = np.zeros((4,))
        if self.in_period('fixation'):
            obs[0] = 1
            if self.actions[int(action)] != 0:
                info['new_trial'] = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if self.actions[int(action)] == 2:
                if trial['wager']:
                    reward = self.rewards['sure']
                    norm_rew =\
                        (reward-self.rewards['fail'])/(self.rewards['correct']-self.rewards['fail'])
                    self.performance = norm_rew
                else:
                    reward = self.rewards['abort']
            else:
                gt_sign = np.sign(trial['ground_truth'])
                action_sign = np.sign(self.actions[int(action)])
                if gt_sign == action_sign:
                    reward = self.rewards['correct']
                    self.performance = 1
                elif gt_sign == -action_sign:
                    reward = self.rewards['fail']
            info['new_trial'] = self.actions[int(action)] != 0

        if self.in_period('delay'):
            obs[0] = 1
        elif self.in_period('stimulus'):
            high = (trial['ground_truth'] > 0) + 1
            low = (trial['ground_truth'] < 0) + 1
            obs[high] = self.scale(+trial['coh']) +\
                self.rng.randn()*self.sigma_dt
            obs[low] = self.scale(-trial['coh']) +\
                self.rng.randn()*self.sigma_dt
        if trial['wager'] and self.in_period('sure'):
            obs[3] = 1

        return obs, reward, False, info


if __name__ == '__main__':
    env = PostDecisionWager()
    env.seed(seed=0)
    ngym.utils.plot_env(env, num_steps_env=100)  # , def_act=0)

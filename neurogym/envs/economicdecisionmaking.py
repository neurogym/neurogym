#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from neurogym.utils import tasktools
import neurogym as ngym
from gym import spaces


class EconomicDecisionMaking(ngym.PeriodEnv):
    r"""Agents choose between two stimuli (A and B; where A is preferred)
    offered in different amounts.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature04676',
        'paper_name': '''Neurons in the orbitofrontal cortex encode
         economic value''',
        'tags': ['perceptual', 'value-based']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        # trial conditions
        self.B_to_A = 1/2.2
        self.juices = [('a', 'b'), ('b', 'a')]
        self.offers = [(0, 1), (1, 3), (1, 2), (1, 1), (2, 1),
                       (3, 1), (4, 1), (6, 1), (2, 0)]

        # Input noise
        sigma = np.sqrt(2*100*0.001)
        self.sigma_dt = sigma/np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +0.22}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 1500),
            'offer_on': ('uniform', [1000, 2000]),
            'decision': ('constant', 750)}
        if timing:
            self.timing.update(timing)

        self.R_B = self.B_to_A * self.rewards['correct']
        self.R_A = self.rewards['correct']
        self.abort = False
        # Increase initial policy -> baseline weights
        self.baseline_Win = 10

        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice1': 1, 'choice2': 2}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7, ),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0,
                        'a1': 1, 'b1': 2,  # a or b for choice 1
                        'a2': 3, 'b2': 4,  # a or b for choice 2
                        'n1': 5, 'n2': 6  # amount for choice 1 or 2
                        }

    def new_trial(self, **kwargs):
        self.trial = {
            'juice': self.juices[self.rng.choice(len(self.juices))],
            'offer': self.offers[self.rng.choice(len(self.offers))]
        }
        self.trial.update(kwargs)

        juice1, juice2 = self.trial['juice']
        n_b, n_a = self.trial['offer']

        if juice1 == 'a':
            n1, n2 = n_a, n_b
        else:
            n1, n2 = n_b, n_a

        self.add_period(['fixation', 'offer_on', 'decision'], after=0, last_period=True)

        self.add_ob(1, ['fixation', 'offer_on'], where='fixation')
        self.add_ob(1, 'offer_on', where=juice1 + '1')
        self.add_ob(1, 'offer_on', where=juice2 + '2')
        self.add_ob(n1/5., 'offer_on', where='n1')
        self.add_ob(n2/5., 'offer_on', where='n2')

    def _step(self, action):
        trial = self.trial

        new_trial = False

        obs = self.ob_now

        reward = 0
        if self.in_period('fixation') or self.in_period('offer_on'):
            if action != self.act_dict['fixation']:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action in [self.act_dict['choice1'], self.act_dict['choice2']]:
                new_trial = True

                juice1, juice2 = trial['juice']

                n_b, n_a = trial['offer']
                r_a = n_a * self.R_A
                r_b = n_b * self.R_B

                if juice1 == 'A':
                    r1, r2 = r_a, r_b
                else:
                    r1, r2 = r_b, r_a

                if action == self.act_dict['choice1']:
                    reward = r1
                    self.performance = r1 > r2
                elif action == self.act_dict['choice2']:
                    reward = r2
                    self.performance = r2 > r1

        return obs, reward, False, {'new_trial': new_trial, 'gt': 0}


if __name__ == '__main__':
    env = EconomicDecisionMaking()
    ngym.utils.plot_env(env, num_steps_env=200, def_act=1)
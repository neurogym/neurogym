"""Multi-Sensory Integration"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym

# TODO: This is not finished yet. Need to compare with original paper
class MultiSensoryIntegration(ngym.TrialEnv):
    r"""Multi-sensory integration."""
    metadata = {
        'description': None,
        'paper_link': None,
        'paper_name': None,
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0,
                 dim_ring=2):
        super().__init__(dt=dt)

        # trial conditions
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 300,
            # 'target': 350,  # TODO: not implemented
            'stimulus': 750,
            # 'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': 100}  # XXX: not specified
        if timing:
            self.timing.update(timing)
        self.abort = False

        # set action and observation space
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + 2 * dim_ring,), dtype=np.float32)
        self.ob_dict = {'fixation': 0,
                        'stimulus_mod1': range(1, dim_ring + 1),
                        'stimulus_mod2': range(dim_ring + 1, 2 * dim_ring + 1)}

        self.action_space = spaces.Discrete(1+dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring+1)}

    def _new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'coh_prop': self.rng.rand(),
        }
        self.trial.update(kwargs)

        coh_0 = self.trial['coh'] * self.trial['coh_prop']
        coh_1 = self.trial['coh'] * (1 - self.trial['coh_prop'])
        ground_truth = self.trial['ground_truth']
        stim_theta = self.theta[ground_truth]

        # Periods
        periods = ['fixation', 'stimulus', 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        stim = np.cos(self.theta - stim_theta) * (coh_0 / 200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus_mod1')
        stim = np.cos(self.theta - stim_theta) * (coh_1 / 200) + 0.5
        self.add_ob(stim, 'stimulus', where='stimulus_mod2')
        self.add_randn(0, self.sigma, 'stimulus')
        self.set_ob(0, 'decision')

        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

    def _step(self, action):
        obs = self.ob_now
        gt = self.gt_now

        new_trial = False
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:  # broke fixation
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}

"""
Context-dependent integration task, based on
  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.
  http://dx.doi.org/10.1038/nature12742

"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


class Mante(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent 
        dynamics in prefrontal cortex''',
        'default_timing': {
            'fixation': ('constant', 750),
            'stimulus': ('constant', 750),
            'delay': ('truncated_exponential', [300, 83, 1200]),
            'decision': ('constant', 500)},
    }

    def __init__(self, dt=100, timing=None):
        super(Mante, self).__init__(dt=dt, timing=timing)

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]

        # Input noise
        self.sigma = np.sqrt(2*100*0.02)
        self.sigma_dt = self.sigma/np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.abort = False

        # set action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.rng.choice(self.contexts),
            'coh_m': self.rng.choice(self.cohs),
            'coh_c': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        choice_m, choice_c = self.trial['ground_truth'], self.trial['other_choice']
        if self.trial['context'] == 1:
            choice_m, choice_c = choice_c, choice_m
        coh_m, coh_c = self.trial['coh_m'], self.trial['coh_c']
        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('delay', after='stimulus')
        self.add_epoch('decision', after='delay', last_epoch=True)

        if choice_m == 1:
            high_m, low_m = 2, 3
        else:
            high_m, low_m = 3, 2

        if choice_c == 1:
            high_c, low_c = 4, 5
        else:
            high_c, low_c = 5, 4

        tmp = np.zeros(6)
        tmp[[high_m, low_m, high_c, low_c]] =\
            (1 + np.array([coh_m, -coh_m, coh_c, -coh_c])/100)/2
        self.set_ob('stimulus', tmp)
        self.obs[:, self.trial['context']] = 1
        self.set_ob('decision', np.zeros(6))
        self.obs[self.stimulus_ind0:self.stimulus_ind1, 2:] += np.random.randn(
            self.stimulus_ind1-self.stimulus_ind0, 4) * self.sigma_dt

        self.set_groundtruth('fixation', 0)
        self.set_groundtruth('stimulus', 0)
        self.set_groundtruth('delay', 0)
        self.set_groundtruth('decision', self.trial['ground_truth'])

    def _step(self, action):
        obs = self.obs_now
        gt = self.gt_now

        new_trial = False
        reward = 0
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:  # broke fixation
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}

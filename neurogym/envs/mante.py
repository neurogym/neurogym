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
        'description': 'Agent has to perform one of two different perceptual discriminations. On every trial, a contextual cue indicates which one to perform.',
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent 
        dynamics in prefrontal cortex''',
        'default_timing': {
            'fixation': ('constant', 300),
            'target': ('constant', 350),  # TODO: not implemented
            'stimulus': ('constant', 750),
            'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': ('constant', 100)}, # XXX: not specified
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)

        # trial conditions
        self.contexts = [1, 2]  # index for context inputs
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
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        choice_0, choice_1 = self.trial['ground_truth'], self.trial['other_choice']
        if self.trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = self.trial['coh_0'], self.trial['coh_1']
        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('delay', after='stimulus')
        self.add_epoch('decision', after='delay', last_epoch=True)

        high_0, low_0 = (3, 4) if choice_0 == 1 else (4, 3)
        high_1, low_1 = (5, 6) if choice_1 == 1 else (6, 5)

        self.obs[:, 0] = 1
        ob = self.view_ob('stimulus')
        ob[:, [high_0, low_0, high_1, low_1]] = (1 + np.array([coh_0, -coh_0, coh_1, -coh_1])/100)/2
        ob[:, 3:] += np.random.randn(ob.shape[0], 4) * self.sigma_dt
        self.set_ob('decision', np.zeros(7))
        self.obs[:, self.trial['context']] = 1

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


from neurogym.inputs import GaussianNoise

# TODO: Under development
class ManteWithAbstraction(ngym.EpochEnv):
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent 
        dynamics in prefrontal cortex''',
        'default_timing': {
            'fixation': ('constant', 300),
            'target': ('constant', 350),  # TODO: not implemented
            'stimulus': ('constant', 750),
            'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': ('constant', 100)}, # XXX: not specified
    }

    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt, timing=timing)

        # trial conditions
        self.contexts = ['context0', 'context1']  # index for context inputs
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
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7),
                                            dtype=np.float32)

        # Add optional way to carve up observation space
        self.locations ={
            'fixation': 0,
            'context0': 1,
            'context1': 2,
            'modality0_input0': 3,
            'modality0_input1': 4,
            'modality1_input0': 5,
            'modality1_input1': 6,
        }

    def new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'other_choice': self.rng.choice(self.choices),
            'context': self.rng.choice(self.contexts),
            'coh_0': self.rng.choice(self.cohs),
            'coh_1': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        choice_0, choice_1 = self.trial['ground_truth'], self.trial['other_choice']
        if self.trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = self.trial['coh_0'], self.trial['coh_1']
        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('delay', after='stimulus')
        self.add_epoch('decision', after='delay', last_epoch=True)

        high_0, low_0 = (3, 4) if choice_0 == 1 else (4, 3)
        high_1, low_1 = (5, 6) if choice_1 == 1 else (6, 5)

        self.add_input(1, loc=0, epoch=['fixation', 'stimulus'])
        for loc, coh in zip([high_0, low_0, high_1, low_1],
                            [coh_0, -coh_0, coh_1, -coh_1]):
            self.add_input(GaussianNoise(mu=(1 + coh/100)/2, sigma=self.sigma_dt),
                           loc=loc, epoch='stimulus')
        self.add_input(1, loc=self.trial['context'], epoch=['fixation', 'stimulus', 'delay'])

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
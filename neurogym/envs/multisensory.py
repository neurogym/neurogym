"""Multi-Sensory Integration"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym

# TODO: This is not finished yet
class MultiSensoryIntegration(ngym.PeriodEnv):
    r"""Multi-sensory integration."""
    metadata = {
        'description': None,
        'paper_link': None,
        'paper_name': None,
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        # trial conditions
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]
        self.coh_diffs = [-50, -15, -5, 0, 5, 15, 50]

        # Input noise
        sigma = np.sqrt(2*100*0.02)
        self.sigma_dt = sigma/np.sqrt(self.dt)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 300),
            # 'target': ('constant', 350),  # TODO: not implemented
            'stimulus': ('constant', 750),
            # 'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)
        self.abort = False

        # set action and observation space
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice1': 1, 'choice2': 2}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,),
                                            dtype=np.float32)
        names = ['fixation', 'stim1_mod1', 'stim2_mod1',
                 'stim1_mod2', 'stim2_mod2']
        self.ob_dict = {name: i for i, name in enumerate(names)}

    def new_trial(self, **kwargs):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'coh_diff': self.rng.choice(self.coh_diffs),
        }
        self.trial.update(kwargs)

        if self.trial['ground_truth'] == 1:
            signed_coh = self.trial['coh']
        else:
            signed_coh = -self.trial['coh']
        signed_coh_0 = (signed_coh + self.trial['coh_diff']) / 2
        signed_coh_1 = (signed_coh - self.trial['coh_diff']) / 2
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'decision']
        self.add_period(periods, after=0, last_period=True)

        self.add_ob(1, where='fixation')
        self.add_ob((1 + signed_coh_0 / 100) / 2, period='stimulus', where='stim1_mod1')
        self.add_ob((1 - signed_coh_0 / 100) / 2, period='stimulus', where='stim2_mod1')
        self.add_ob((1 + signed_coh_1 / 100) / 2, period='stimulus', where='stim1_mod2')
        self.add_ob((1 - signed_coh_1 / 100) / 2, period='stimulus', where='stim2_mod2')
        self.add_randn(0, self.sigma_dt, 'stimulus')
        self.set_ob(0, 'decision')

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

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

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = MultiSensoryIntegration()
    env.seed(seed=0)
    ngym.utils.plot_env(env, num_steps_env=100, def_act=0)

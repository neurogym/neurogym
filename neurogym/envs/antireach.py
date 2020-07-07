"""Anti-reach or anti-saccade task."""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.utils import tasktools


class AntiReach(ngym.PeriodEnv):
    """Anti-response task.

    The agent has to move in the direction opposite to the one indicated
    by the observation.
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nrn1345',
        'paper_name': """Look away: the anti-saccade task and
        the voluntary control of eye movement""",
        'tags': ['perceptual', 'steps action space']
    }

    def __init__(self, dt=100, anti=True, rewards=None, timing=None,
                 dim_ring=32):
        super().__init__(dt=dt)

        self.anti = anti

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 500),
            'stimulus': ('constant', 500),
            'delay': ('constant', 0),
            'decision': ('constant', 500)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.dim_ring = dim_ring
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.choices = np.arange(dim_ring)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1+dim_ring,), dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': range(1, dim_ring + 1)}

        self.action_space = spaces.Discrete(1+dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring + 1)}

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'anti': self.anti,
        }
        self.trial.update(kwargs)

        ground_truth = self.trial['ground_truth']
        if self.trial['anti']:
            stim_theta = np.mod(self.theta[ground_truth] + np.pi, 2*np.pi)
        else:
            stim_theta = self.theta[ground_truth]

        # Periods
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods, last_period=True)

        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        stim = np.cos(self.theta - stim_theta)
        self.add_ob(stim, 'stimulus', where='stimulus')

        self.set_groundtruth(self.act_dict['choice'][ground_truth], 'decision')

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


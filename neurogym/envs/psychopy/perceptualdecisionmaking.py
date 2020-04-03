"""Random dot motion task."""

import numpy as np
from gym import spaces
from psychopy import visual

import neurogym as ngym
from .psychopy_env import PsychopyEnv


class PerceptualDecisionMaking(PsychopyEnv):
    """Two-alternative forced choice task in which the subject has to
    integrate two stimuli to decide which one is higher on average.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        dim_ring: int, dimension of ring input and output
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/12/12/4745',
        'paper_name': '''The analysis of visual motion: a comparison of
        neuronal and psychophysical performance''',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None,
                 stim_scale=1., dim_ring=2, win_size=(100, 100)):
        super().__init__(dt=dt, win_size=win_size)
        # The strength of evidence, modulated by stim_scale
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('constant', 100),  # TODO: depends on subject
            'stimulus': ('constant', 2000),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)

        self.action_space = spaces.Discrete(1 + dim_ring)
        self.act_dict = {'fixation': 0, 'choice': range(1, dim_ring + 1)}

    def new_trial(self, **kwargs):
        # Trial info
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
        }
        self.trial.update(kwargs)

        coh = self.trial['coh']
        ground_truth = self.trial['ground_truth']
        stim_theta = self.theta[ground_truth] * (180/np.pi)

        # Periods
        self.add_period(['fixation', 'stimulus', 'decision'], after=0,
                        last_period=True)

        # Observations
        stim = visual.DotStim(self.win, nDots=30, dotSize=1, speed=0.05,
                              dotLife=10, signalDots='same',
                              fieldShape='circle', coherence=(coh/100),
                              dir=stim_theta * (180/np.pi))
        self.add_ob(stim, 'stimulus')

        # Ground truth
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

"""Multi-Sensory Integration"""
from __future__ import division

import numpy as np
from gym import spaces
import neurogym as ngym

# TODO: This is not implemented yet
class MultiSensoryIntegration(ngym.PeriodEnv):
    metadata = {
        'description': 'Agent has to perform one of two different perceptual' +
        ' discriminations. On every trial, a contextual cue indicates which' +
        ' one to perform.',
        'paper_link': 'https://www.nature.com/articles/nature12742',
        'paper_name': '''Context-dependent computation by recurrent
         dynamics in prefrontal cortex''',
        'tags': ['perceptual', 'context dependent', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        Agent has to perform one of two different perceptual discriminations.
        On every trial, a contextual cue indicates which one to perform.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
        timing: Description and duration of periods forming a trial.
        """
        super().__init__(dt=dt)

        # trial conditions
        self.contexts = [1, 2]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]

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
            'delay': ('truncated_exponential', [600, 300, 3000]),
            'decision': ('constant', 100)}  # XXX: not specified
        if timing:
            self.timing.update(timing)
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

        choice_0, choice_1 =\
            self.trial['ground_truth'], self.trial['other_choice']
        if self.trial['context'] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = self.trial['coh_0'], self.trial['coh_1']
        # -----------------------------------------------------------------------
        # Periods
        # -----------------------------------------------------------------------
        periods = ['fixation', 'stimulus', 'delay', 'decision']
        self.add_period(periods, after=0, last_period=True)

        high_0, low_0 = (3, 4) if choice_0 == 1 else (4, 3)
        high_1, low_1 = (5, 6) if choice_1 == 1 else (6, 5)

        self.obs[:, 0] = 1
        ob = self.view_ob('stimulus')
        ob[:, [high_0, low_0, high_1, low_1]] =\
            (1 + np.array([coh_0, -coh_0, coh_1, -coh_1])/100)/2
        ob[:, 3:] += self.rng.randn(ob.shape[0], 4) * self.sigma_dt
        self.set_ob(np.zeros(7), 'decision')
        self.obs[:, self.trial['context']] = 1

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def _step(self, action):
        obs = self.obs_now
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

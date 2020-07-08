# -*- coding: utf-8 -*-

import numpy as np
from gym import spaces
import neurogym as ngym


class DelayComparison(ngym.TrialEnv):
    """Delay comparison.

    Two-alternative forced choice task in which the subject
    has to compare two stimuli separated by a delay to decide
    which one has a higher frequency.
    """
    metadata = {
        'paper_link': 'https://www.jneurosci.org/content/30/28/9424',
        'paper_name': '''Neuronal Population Coding of Parametric
        Working Memory''',
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0):
        super().__init__(dt=dt)

        # trial conditions
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': ('uniform', (1500, 3000)),
            'f1': ('constant', 500),
            'delay': ('constant', 3000),
            'f2': ('constant', 500),
            'decision': ('constant', 100)}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Input scaling
        self.fall = np.ravel(self.fpairs)
        self.fmin = np.min(self.fall)
        self.fmax = np.max(self.fall)

        # action and observation space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)
        self.ob_dict = {'fixation': 0, 'stimulus': 1}
        self.action_space = spaces.Discrete(3)
        self.act_dict = {'fixation': 0, 'choice': [1, 2]}

    def _new_trial(self, **kwargs):
        self.trial = {
            'ground_truth': self.rng.choice(self.act_dict['choice']),
            'fpair': self.fpairs[self.rng.choice(len(self.fpairs))]
        }
        self.trial.update(kwargs)

        f1, f2 = self.trial['fpair']
        if self.trial['ground_truth'] == 2:
            f1, f2 = f2, f1
        self.trial['f1'] = f1
        self.trial['f2'] = f2

        # Periods
        periods = ['fixation', 'f1', 'delay', 'f2', 'decision']
        self.add_period(periods)

        self.add_ob(1, where='fixation')
        self.add_ob(self.scale_p(f1), 'f1', where='stimulus')
        self.add_ob(self.scale_p(f2), 'f2', where='stimulus')
        self.set_ob(0, 'decision')
        self.add_randn(0, self.sigma, ['f1', 'f2'])

        self.set_groundtruth(self.trial['ground_truth'], 'decision')

    def scale(self, f):
        return (f - self.fmin)/(self.fmax - self.fmin)

    def scale_p(self, f):
        return (1 + self.scale(f))/2

    def scale_n(self, f):
        return (1 - self.scale(f))/2

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']

        return ob, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = DelayComparison()
    ngym.utils.plot_env(env, num_steps=100)

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:00:32 2019

@author: MOLANO

A parametric working memory task, based on

  Neuronal population coding of parametric working memory.
  O. Barak, M. Tsodyks, & R. Romo, JNS 2010.

  http://dx.doi.org/10.1523/JNEUROSCI.1875-10.2010

"""
import numpy as np
from gym import spaces
import neurogym as ngym


class DelayedComparison(ngym.PeriodEnv):
    metadata = {
        'description': """Two-alternative forced choice task in which
         the subject has to compare two stimuli separated by a delay
         to decide which one has a higher frequency.""",
        'paper_link': 'https://www.jneurosci.org/content/30/28/9424',
        'paper_name': '''Neuronal Population Coding of Parametric
        Working Memory''',
        'timing': {
            'fixation': ('uniform', (1500, 3000)),
            'f1': ('constant', 500),
            'delay': ('constant', 3000),
            'f2': ('constant', 500),
            'decision': ('constant', 100)},  # XXX: not specified
        'tags': ['perceptual', 'working memory', 'two-alternative',
                 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None):
        """
        Two-alternative forced choice task in which the subject
        has to compare two stimuli separated by a delay to decide
        which one has a higher frequency.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: 0., float)
        timing: Description and duration of periods forming a trial.
        """
        # call ngm __init__ function
        super().__init__(dt=dt, timing=timing)

        # trial conditions
        self.choices = [1, 2]
        self.fpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]

        # Input noise
        sigma = np.sqrt(2*100*0.001)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        reward_default = {'R_ABORTED': -0.1, 'R_CORRECT': +1.,
                          'R_FAIL': 0.}
        if rewards is not None:
            reward_default.update(rewards)
        self.R_ABORTED = reward_default['R_ABORTED']
        self.R_CORRECT = reward_default['R_CORRECT']
        self.R_FAIL = reward_default['R_FAIL']

        self.abort = False

        # Input scaling
        self.fall = np.ravel(self.fpairs)
        self.fmin = np.min(self.fall)
        self.fmax = np.max(self.fall)

        # action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2, ),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'fpair': self.rng.choice(self.fpairs)
        }
        self.trial.update(kwargs)
        f1, f2 = self.trial['fpair']
        if self.trial['ground_truth'] == 2:
            f1, f2 = f2, f1
        # -------------------------------------------------------------------------
        # Periods
        # --------------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('f1', after='fixation')
        self.add_period('delay', after='f1')
        self.add_period('f2', after='delay')
        self.add_period('decision', after='f2', last_period=True)

        self.set_ob('fixation', [1, 0])
        self.set_ob('f1', [1, self.scale_p(f1)])
        self.set_ob('delay', [1, 0])
        self.set_ob('f2', [1, self.scale_p(f2)])
        self.set_ob('decision', [0, 0])
        ob = self.view_ob('f1')
        ob[:, 1] += np.random.randn(ob.shape[0]) * self.sigma_dt
        ob = self.view_ob('f2')
        ob[:, 1] += np.random.randn(ob.shape[0]) * self.sigma_dt

        self.set_groundtruth('decision', self.trial['ground_truth'])

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
        obs = self.obs_now
        # rewards
        reward = 0
        if self.in_period('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                    self.performance = 1
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = DelayedComparison()
    ngym.utils.plot_env(env, num_steps_env=50000)

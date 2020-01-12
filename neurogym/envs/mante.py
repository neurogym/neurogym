"""
Context-dependent integration task, based on
  Context-dependent computation by recurrent dynamics in prefrontal cortex.
  V Mante, D Sussillo, KV Shinoy, & WT Newsome, Nature 2013.
  http://dx.doi.org/10.1038/nature12742

Code adapted from github.com/frsong/pyrl
"""
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.ops import tasktools
import neurogym as ngym


class Mante(ngym.EpochEnv):
    def __init__(self, dt=100, timing=[750, 750, 83, 300, 1200, 500]):
        # call ngm __init__ function
        super().__init__(dt=dt)

        # Inputs
        self.inputs = tasktools.to_map('motion', 'color',
                                       'm-left', 'm-right',
                                       'c-left', 'c-right')
        # Actions
        self.actions = tasktools.to_map('FIXATE', 'left', 'right')

        # trial conditions
        self.contexts = [0, 1]  # index for context inputs
        self.choices = [1, 2]  # left, right choice
        self.cohs = [5, 15, 50]

        # Input noise
        self.sigma = np.sqrt(2*100*0.02)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_MISS = 0.
        self.abort = False

        # Epoch durations
        self.fixation = timing[0]
        self.stimulus = timing[1]
        self.delay_min = timing[2]
        self.delay_mean = timing[3]
        self.delay_max = timing[4]
        self.decision = timing[5]
        self.mean_trial_duration = self.fixation + self.stimulus +\
            self.delay_mean + self.decision

        # set action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,),
                                            dtype=np.float32)

    def __str__(self):
        string = 'mean trial duration: ' + str(self.mean_trial_duration) + '\n'
        string += ' (max num. steps: ' + str(self.mean_trial_duration / self.dt)
        return string

    def _new_trial(self):
        # -------------------------------------------------------------------------
        # Trial
        # -------------------------------------------------------------------------
        context = self.rng.choice(self.contexts)
        choice_m = self.rng.choice(self.choices)
        choice_c = self.rng.choice(self.choices)
        coh_m = self.rng.choice(self.cohs)
        coh_c = self.rng.choice(self.cohs)

        ground_truth = choice_m if context == 0 else choice_c

        # -----------------------------------------------------------------------
        # Epochs
        # -----------------------------------------------------------------------
        delay = self.delay_min +\
            tasktools.trunc_exp(self.rng, self.dt, self.delay_mean,
                                            xmax=self.delay_max)
        self.add_epoch('fixation', self.fixation, start=0)
        self.add_epoch('stimulus', self.stimulus, after='fixation')
        self.add_epoch('delay', delay, after='stimulus')
        self.add_epoch('decision', self.decision, after='delay', last_epoch=True)

        if choice_m == 1:
            high_m, low_m = 2, 3
        else:
            high_m, low_m = 3, 2

        if choice_c == 1:
            high_c, low_c = 4, 5
        else:
            high_c, low_c = 5, 4

        tmp = np.zeros(6)
        tmp[[high_m, low_m, high_c, low_c]] = (1 + np.array([coh_m, -coh_m, coh_c, -coh_c])/100)/2
        self.set_ob('stimulus', tmp)
        self.obs[:, context] = 1
        self.set_ob('decision', np.zeros(6))
        self.obs[self.stimulus_ind0:self.stimulus_ind1, 2:] += np.random.randn(
            self.stimulus_ind1-self.stimulus_ind0, 4) * (self.sigma/np.sqrt(self.dt))

        return {
            'context': context,
            'choice_m': choice_m,
            'choice_c': choice_c,
            'coh_m': coh_m,
            'coh_c': coh_c,
            'ground_truth': ground_truth
            }

    def _step(self, action):
        # -----------------------------------------------------------------
        # Reward
        # -----------------------------------------------------------------
        trial = self.trial

        info = {'new_trial': False}
        info['gt'] = np.zeros((3,))
        reward = 0
        if self.in_epoch('fixation'):
            info['gt'][0] = 1
            if (action != self.actions['FIXATE']):
                info['new_trial'] = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            info['gt'][trial['ground_truth']] = 1
            if action in [1, 2]:  # action taken
                info['new_trial'] = True
                if action == trial['ground_truth']:
                    reward = self.R_CORRECT
        else:
            info['gt'][0] = 1

        obs = self.obs[self.t_ind]

        return obs, reward, False, info


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:00:26 2020

@author: martafradera
"""

import numpy as np
from gym import spaces
import neurogym as ngym
from neurogym.meta import tasks_info


class SvsN(ngym.EpochEnv):
    metadata = {
            'description': '',
            'paper_link': '',
            'paper_name': '',
            'timing': {
                    'fixation': ('constant', 500),
                    'noise1': ('truncated_exponential', [1500, 0, 3000]),
                    'stimulus': ('truncated_exponential', [1000, 500, 1500]),
                    'noise2': ('truncated_exponential', [1500, 0, 3000]),
                    'decision': ('constant', 500)}
            }

    def __init__(self, dt=100, timing=None, extra_input_param=None):
        super().__init__(dt=dt, timing=timing)
        # Possible decisions at the end of the trial
        self.choices = [0, 1]

        # Noise added to the observations
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1  # reward given when break fixation
        self.R_CORRECT = +1.  # reward given when correct
        self.R_FAIL = 0.  # reward given when incorrect
        self.R_MISS = 0.  # reward given when not responding
        # whether to abort (T) or not (F) the trial when breaking fixation:
        self.abort = False
        # action and observation spaces: [fixate, go]
        self.action_space = spaces.Discrete(2)
        # observation space: [fixation cue, stim]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        Here you have to set (at least):
        1. The ground truth: the correct answer for the created trial.
        2. The trial periods: fixation, stimulus...
            """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.trial = {'ground_truth': self.rng.choice(self.choices)}
        self.trial.update(kwargs)  # allows wrappers to modify the trial
        ground_truth = self.trial['ground_truth']
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('noise1', after='fixation')
        self.add_epoch('stimulus', after='noise1')
        self.add_epoch('noise2', after='stimulus')
        self.add_epoch('decision', after='noise2', last_epoch=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0])
        # NOISE:
        noise1 = self.view_ob('noise1')
        noise1[:, 1:] += np.random.randn(noise1.shape[0], 1) * self.sigma_dt
        noise2 = self.view_ob('noise2')
        noise2[:, 1:] += np.random.randn(noise2.shape[0], 1) * self.sigma_dt
        # STIMULUS
        # view_ob return a pointer to observations during stimulus period:
        stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
        # SET THE STIMULUS
        # adding gaussian noise to stimulus with std = self.sigma_dt
        if ground_truth == 1:
            stimulus[:, 1:] +=\
                0.5 + np.random.randn(stimulus.shape[0], 1) * self.sigma_dt
        else:
            stimulus[:, 1:] +=\
                np.random.randn(stimulus.shape[0], 1) * self.sigma_dt
        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------
        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # Example structure
        if self.in_epoch('fixation'):  # during fixation period
            if action != 0:  # if fixation break
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):  # during decision period
            if action != 0:
                new_trial = True
                if action == gt:  # if correct
                    reward = self.R_CORRECT
                else:  # if incorrect
                    reward = self.R_FAIL

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = SvsN()
    tasks_info.plot_struct(env)

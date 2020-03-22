#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 10:09:00 2020

@author: manuel
"""

import numpy as np
from gym import spaces
import neurogym as ngym


class YourTask(ngym.PeriodEnv):  # TIP: if task has periods (alt.: ngym.TrialEnv)
    metadata = {
        'description': '',
        'paper_link': '',
        'paper_name': '',
        # alts.: 'constant', 'uniform', 'truncated_exponential' and 'choice'
        # (see neurogym/utils/tasktools.random_number_fn)
        self.timing = {
            'period 1 (e.g. fixation)': ('constant', 'value'),
            'period 2 (e.g. stimulus)': ('truncated_exponential',
                                         ['mean', 'min', 'max']),
            'period 3 (e.g. decision)': ('uniform', ('min', 'max'))},
        'extra input parameter 1': 'value',
        'extra input parameter 2': 'value',
        'tags': ['property 1', 'property 2']
    }

    def __init__(self, dt=100, timing=None, extra_input_param=None):
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = [1, 2]  # e.g. [left, right]

        # Noise added to the observations
        sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma = sigma / np.sqrt(self.dt)

        # Rewards
        self.rewards['abort'] = -0.1  # reward given when break fixation
        self.rewards['correct'] = +1.  # reward given when correct
        self.rewards['fail'] = 0.  # reward given when incorrect
        # whether to abort (T) or not (F) the trial when breaking fixation:
        self.abort = False
        # action and observation spaces: [fixate, got left, got right]
        self.action_space = spaces.Discrete(3)
        # observation space: [fixation cue, left stim, right stim]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
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
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation')
        self.add_period('decision', after='stimulus', last_period=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0, 0])
        # STIMULUS
        # view_ob return a pointer to observations during stimulus period:
        stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
        # SET THE STIMULUS
        # adding gaussian noise to stimulus with std = self.sigma
        stimulus[:, 1:] +=\
            np.random.randn(stimulus.shape[0], 2) * self.sigma
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
        if self.in_period('fixation'):  # during fixation period
            if action != 0:  # if fixation break
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('decision'):  # during decision period
            if action != 0:
                new_trial = True
                if action == gt:  # if correct
                    reward = self.rewards['correct']
                else:  # if incorrect
                    reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}

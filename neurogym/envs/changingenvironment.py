#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:47:15 2020

@author: martafradera
"""

import numpy as np
from gym import spaces
import neurogym as ngym
from neurogym.meta import tasks_info


class CE(ngym.EpochEnv):
    metadata = {
        'description': '',
        'paper_link': 'https://www.pnas.org/content/113/31/E4531',
        'paper_name': '''Hierarchical decision processes that operate over
        distinct timescales underlie choice and changes in strategy''',
        'timing': {
            'fixation': ('constant', 500),
            'stimulus': ('truncated_exponential', [1000, 500, 1500]),
            'decision': ('constant', 500)},
        'stimEv': 'Controls the difficulty of the experiment. (def: 1.)',
        'cxt_ch_prob': 'Probability of changing context.',
        'rep_prob': '''Specifies probabilities of repeating for each block.
        (def: (.2, .8))''',
    }

    def __init__(self, dt=100, timing=None, stimEv=1., cxt_ch_prob=0.2,
                 rep_prob=(.2, .8), cxt_cue=False):
        super().__init__(dt=dt, timing=timing)

        # Possible contexts
        self.cxt_ch_prob = cxt_ch_prob
        self.curr_cxt = 0
        self.rep_prob = rep_prob
        self.cxt_cue = cxt_cue

        # Possible decisions at the end of the trial
        self.choices = [1, 2]  # [left, right]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stimEv

        # Noise added to the observations
        sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = sigma / np.sqrt(self.dt)

        # Rewards
        self.R_ABORTED = -0.1  # reward given when break fixation
        self.R_CORRECT = +1.  # reward given when correct
        self.R_FAIL = 0.  # reward given when incorrect
        # whether to abort (T) or not (F) the trial when breaking fixation:
        self.abort = False
        # action and observation spaces: [fixate, got left, got right]
        self.action_space = spaces.Discrete(5)
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

        if self.rng.random() < self.cxt_ch_prob:
                self.curr_cxt = (self.curr_cxt + 1) % len(self.rep_prob)

        side = self.rng.choice(self.choices)

        self.trial = {
            'side': side,
            'ground_truth': (side + 2 * self.curr_cxt),
            'coh': self.rng.choice(self.cohs),
        }

        self.trial.update(kwargs)  # allows wrappers to modify the trial

        coh = self.trial['coh']
        ground_truth = self.trial['ground_truth']

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        self.add_epoch('fixation', after=0)
        self.add_epoch('stimulus', after='fixation')
        self.add_epoch('decision', after='stimulus', last_epoch=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0, 0])
        # STIMULUS
        # view_ob return a pointer to observations during stimulus period:
        stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
        stimulus[:, 1:] = (1 - coh / 100) / 2
        # coh for correct side
        stimulus[:, side] = (1 + coh / 100) / 2
        # adding gaussian noise to stimulus with std = self.sigma_dt
        stimulus[:, 1:] +=\
            np.random.randn(stimulus.shape[0], 2) * self.sigma_dt
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
        obs = self.obs_now
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

        if self.cxt_cue:
            cue = np.array([self.curr_cxt])
            obs = np.concatenate((cue, obs), axis=0)

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = CE(cxt_cue=True)
    tasks_info.plot_struct(env)a
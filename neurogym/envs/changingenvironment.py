#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:47:15 2020

@author: martafradera
"""

import numpy as np
from gym import spaces
import neurogym as ngym
from neurogym.meta import info


class ChangingEnvironment(ngym.PeriodEnv):
    metadata = {
        'description': 'Random Dots Motion tasks in which the correct action' +
        ' depends on a randomly changing context',
        'paper_link': 'https://www.pnas.org/content/113/31/E4531',
        'paper_name': '''Hierarchical decision processes that operate over
        distinct timescales underlie choice and changes in strategy''',
        'timing': {
            'fixation': ('constant', 500),
            'stimulus': ('truncated_exponential', [1000, 500, 1500]),
            'decision': ('constant', 500)},
        'stimEv': 'Controls the difficulty of the experiment. (def: 1.)',
        'cxt_ch_prob': 'Probability of changing context.',
        'cxt_cue': 'Whether to show context as a cue.',
        'tags': ['perceptual', '2-alternative', 'supervised',
                 'context dependent']
    }

    def __init__(self, dt=100, timing=None, stimEv=1., cxt_ch_prob=0.01,
                 cxt_cue=False):
        super().__init__(dt=dt, timing=timing)

        # Possible contexts
        self.cxt_ch_prob = cxt_ch_prob
        self.curr_cxt = self.rng.choice([0, 1])
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
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(3+1*cxt_cue,),
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
            self.curr_cxt = 1*(not self.curr_cxt)

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
        if not self.cxt_cue:
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
        else:
            self.set_ob('fixation', [self.curr_cxt, 0, 0, 0])
            self.set_ob('stimulus', [self.curr_cxt, 0, 0, 0])
            self.set_ob('decision', [self.curr_cxt, 0, 0, 0])
            self.set_ob('fixation', [0, 1, 0, 0])
            # STIMULUS
            # view_ob return a pointer to observations during stimulus period:
            stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
            stimulus[:, 2:] = (1 - coh / 100) / 2
            # coh for correct side
            stimulus[:, side] = (1 + coh / 100) / 2
            # adding gaussian noise to stimulus with std = self.sigma_dt
            stimulus[:, 2:] +=\
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
        if self.in_period('fixation'):  # during fixation period
            if action != 0:  # if fixation break
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):  # during decision period
            if action != 0:
                new_trial = True
                if action == gt:  # if correct
                    reward = self.R_CORRECT
                else:  # if incorrect
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt,
                                    'context': self.curr_cxt}


if __name__ == '__main__':
    env = ChangingEnvironment(cxt_ch_prob=0.05, stimEv=100, cxt_cue=False)
    info.plot_struct(env)

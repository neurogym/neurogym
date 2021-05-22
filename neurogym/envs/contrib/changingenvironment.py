#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:47:15 2020

@author: martafradera
"""

import numpy as np
from gym import spaces
import neurogym as ngym


# TODO: Need a more intuitive name
class ChangingEnvironment(ngym.TrialEnv):
    r"""Random Dots Motion tasks in which the correct action
    depends on a randomly changing context.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        cxt_ch_prob: Probability of changing context. (def: 0.01, float)
        cxt_cue: Whether to show context as a cue. (def: False, bool)
    """
    metadata = {
        'paper_link': 'https://www.pnas.org/content/113/31/E4531',
        'paper_name': '''Hierarchical decision processes that operate
        over distinct timescales underlie choice and changes in strategy''',
        'tags': ['perceptual', 'two-alternative', 'supervised',
                 'context dependent']
    }

    def __init__(self, dt=100, rewards=None, timing=None, stim_scale=1.,
                 sigma=1.0, cxt_ch_prob=0.001, cxt_cue=False):
        super().__init__(dt=dt)

        # Possible contexts
        self.cxt_ch_prob = cxt_ch_prob
        self.curr_cxt = self.rng.choice([0, 1])
        self.cxt_cue = cxt_cue

        # Possible decisions at the end of the trial
        self.choices = [1, 2]  # [left, right]
        # cohs specifies the amount of evidence (which is modulated by stim_scale)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 500,
            'stimulus': ngym.random.TruncExp(1000, 500, 1500),
            'decision': 500}
        if timing:
            self.timing.update(timing)

        # whether to abort (T) or not (F) the trial when breaking fixation:
        self.abort = False
        # action and observation spaces: [fixate, got left, got right]
        self.action_space = spaces.Discrete(5)
        # observation space: [fixation cue, left stim, right stim]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(3+1*cxt_cue,),
                                            dtype=np.float32)

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        if self.rng.rand() < self.cxt_ch_prob:
            self.curr_cxt = 1*(not self.curr_cxt)

        side = self.rng.choice(self.choices)

        trial = {
            'side': side,
            'ground_truth': (side + 2 * self.curr_cxt),
            'coh': self.rng.choice(self.cohs),
        }

        trial.update(kwargs)  # allows wrappers to modify the trial

        coh = trial['coh']
        ground_truth = trial['ground_truth']

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period(['fixation', 'stimulus', 'decision'])
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        if not self.cxt_cue:
            self.set_ob([1, 0, 0], 'fixation')
            # STIMULUS
            # view_ob return a pointer to observations during stimulus period:
            stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
            stimulus[:, 1:] = (1 - coh / 100) / 2
            # coh for correct side
            stimulus[:, side] = (1 + coh / 100) / 2
            # adding gaussian noise to stimulus with std = self.sigma
            stimulus[:, 1:] +=\
                self.rng.randn(stimulus.shape[0], 2) * self.sigma
        else:
            self.set_ob([self.curr_cxt, 0, 0, 0], 'fixation')
            self.set_ob([self.curr_cxt, 0, 0, 0], 'stimulus')
            self.set_ob([self.curr_cxt, 0, 0, 0], 'decision')
            self.set_ob([0, 1, 0, 0], 'fixation')  # TODO: Is this right?
            # STIMULUS
            # view_ob return a pointer to observations during stimulus period:
            stimulus = self.view_ob('stimulus')  # (shape = time x obs-dim)
            stimulus[:, 2:] = (1 - coh / 100) / 2
            # coh for correct side
            stimulus[:, side] = (1 + coh / 100) / 2
            # adding gaussian noise to stimulus with std = self.sigma
            stimulus[:, 2:] +=\
                self.rng.randn(stimulus.shape[0], 2) * self.sigma

        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------
        self.set_groundtruth(ground_truth, 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        obs = self.ob_now
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
                    self.performance = 1
                else:  # if incorrect
                    reward = self.rewards['fail']

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt,
                                    'context': self.curr_cxt}


if __name__ == '__main__':
    env = ChangingEnvironment(cxt_ch_prob=0.05, stim_scale=100, cxt_cue=False)
    ngym.utils.plot_env(env)

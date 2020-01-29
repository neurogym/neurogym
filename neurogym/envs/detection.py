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


class Detection(ngym.EpochEnv):
    metadata = {
            'description': 'The agent has to GO if a stimulus is presented.',
            'paper_link': None,
            'paper_name': None,
            'timing': {
                    'fixation': ('constant', 500),
                    'stimulus': ('truncated_exponential', [1000, 500, 1500])},
            'noise': 'Standard deviation of background noise. (def: 1)',
            'delay': 'If not None indicates the delay, from the moment of' +
            ' the start of the stimulus period when the actual ' +
            'stimulus is presented. Otherwise, the delay is drawn from' +
            ' a uniform distribution. (def: None)',
            'stim_dur': 'Stimulus duration. (def: 100, ms)'
            }

    def __init__(self, dt=100, timing=None, noise=1., delay=None,
                 stim_dur=100):
        super().__init__(dt=dt, timing=timing)
        # Possible decisions at the end of the trial
        self.choices = [0, 1]

        # Noise added to the observations
        self.sigma_dt = noise / np.sqrt(self.dt)
        self.delay = delay
        self.stim_dur = int(stim_dur/self.dt)
        assert self.stim_dur > 0, 'Stimulus duration shorter than dt'
        # Rewards
        self.R_ABORTED = -0.1  # reward given when break fixation
        self.R_CORRECT = +1.  # reward given when correct
        self.R_FAIL = -1.  # reward given when incorrect
        self.R_MISS = -0.5  # reward given when missing the stimulus
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
        self.add_epoch('stimulus', after='fixation', last_epoch=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0])
        # stimulus:
        stim = self.view_ob('stimulus')
        stim[:, 1:] += np.random.randn(stim.shape[0], 1) * self.sigma_dt
        # delay
        # SET THE STIMULUS
        # adding gaussian noise to stimulus with std = self.sigma_dt
        if ground_truth == 1:
            if self.delay is None:
                delay = self.rng.randint(0, stim.shape[0]-self.stim_dur)
            else:
                delay = self.delay
            stim[delay:delay + self.stim_dur, 1] += 0.5
            self.r_tmax = self.R_MISS
        else:
            stim[:, 1:] +=\
                np.random.randn(stim.shape[0], 1) * self.sigma_dt
            delay = 0
            self.r_tmax = 0
        self.delay_trial = delay*self.dt
        # ---------------------------------------------------------------------
        # Ground truth
        # ---------------------------------------------------------------------
        if ground_truth == 1:
            dec = self.view_groundtruth('stimulus')
            dec[delay:] = 1

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
        elif self.in_epoch('stimulus'):  # during stimulus period
            if action != 0:
                new_trial = True
                if ((action == self.trial['ground_truth']) and
                   (self.t >= self.end_t['fixation'] + self.delay_trial)):
                    reward = self.R_CORRECT
                else:  # if incorrect
                    reward = self.R_FAIL

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = Detection(noise=0, timing={'stimulus': ('constant', 200)})
    tasks_info.plot_struct(env, num_steps_env=50,
                           n_stps_plt=50, legend=False)  # ,def_act=1)

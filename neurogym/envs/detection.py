#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:00:26 2020

@author: martafradera
"""

import numpy as np
from gym import spaces
import neurogym as ngym
import warnings


class Detection(ngym.PeriodEnv):
    metadata = {
            'description': 'The agent has to GO if a stimulus is presented.',
            'paper_link': None,
            'paper_name': None,
            'timing': {
                    'fixation': ('constant', 500),
                    'stimulus': ('truncated_exponential', [1000, 500, 1500])},
            'tags': ['perceptual', 'reaction time', 'go-no-go',
                     'supervised']
            }

    def __init__(self, dt=100, rewards=None, timing=None, noise=.5, delay=None,
                 stim_dur=100):
        """
        The agent has to GO if a stimulus is presented.
        dt: Timestep duration. (def: 100 (ms), int)
        rewards:
            R_ABORTED: given when breaking fixation. (def: -0.1, float)
            R_CORRECT: given when correct. (def: +1., float)
            R_FAIL: given when incorrect. (def: -1., float)
            R_MISS:  given when not responding when a response was expected.
            (def: -0.5, float)
        timing: Description and duration of periods forming a trial.
        noise: Standard deviation of background noise. (def: 1., float)
        delay: If not None indicates the delay, from the moment of the start of
        the stimulus period when the actual stimulus is presented. Otherwise,
        the delay is drawn from a uniform distribution. (def: None (ms), int)
        stim_dur: Stimulus duration. (def: 100 (ms), int)
        """
        super().__init__(dt=dt, timing=timing)
        # Possible decisions at the end of the trial
        self.choices = [0, 1]

        # Noise added to the observations
        self.sigma_dt = noise / np.sqrt(self.dt)
        self.delay = delay
        self.stim_dur = int(stim_dur/self.dt)  # in steps should be greater
        # than 1 stp else it wont have enough time to respond within the window
        if self.stim_dur == 1:
            self.extra_step = 1
            if delay is None:
                warnings.warn('Added an extra stp after the actual stimulus,' +
                              ' else model will not be able to respond ' +
                              'within response window (stimulus epoch)')
        else:
            self.extra_step = 0

        if self.stim_dur < 1:
            warnings.warn('Stimulus duration shorter than dt')

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': -1., 'miss': -1}
        if rewards:
            self.rewards.update(rewards)

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
        # Periods
        # ---------------------------------------------------------------------
        self.add_period('fixation', after=0)
        self.add_period('stimulus', after='fixation', last_period=True)
        # ---------------------------------------------------------------------
        # Observations
        # ---------------------------------------------------------------------
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob('fixation', [1, 0])
        # stimulus:
        stim = self.view_ob('stimulus')
        stim[:, 1:] += self.rng.randn(stim.shape[0], 1) * self.sigma_dt
        # delay
        # SET THE STIMULUS
        # adding gaussian noise to stimulus with std = self.sigma_dt
        if ground_truth == 1:
            if self.delay is None:
                # there must be a step after the stim or the model will not
                # be able to respond on time
                max_delay = stim.shape[0]-self.stim_dur-self.extra_step
                delay = 0 if max_delay == 0 else self.rng.randint(0, max_delay)
            else:
                delay = self.delay
            stim[delay:delay + self.stim_dur, 1] += 0.5  # actual stim
            self.r_tmax = self.rewards['miss']
            self.performance = 0
        else:
            stim[:, 1:] +=\
                self.rng.randn(stim.shape[0], 1) * self.sigma_dt
            delay = 0
            self.r_tmax = 0  # response omision is correct but not rewarded
            self.performance = 1

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
        if self.in_period('fixation'):  # during fixation period
            if action != 0:  # if fixation break
                new_trial = self.abort
                reward = self.rewards['abort']
        elif self.in_period('stimulus'):  # during stimulus period
            if action != 0:  # original
                new_trial = True
                if ((action == self.trial['ground_truth']) and
                   (self.t >= self.end_t['fixation'] + self.delay_trial)):
                    reward = self.rewards['correct']
                    self.performance = 1
                else:  # if incorrect
                    reward = self.rewards['fail']
                    self.performance = 0

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    env = Detection(noise=0, timing={'stimulus': ('constant', 200)})
    ngym.utils.plot_env(env, num_steps_env=50,
                        legend=False)  # ,def_act=1)

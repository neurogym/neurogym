#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:39:17 2019

@author: molano


General two-alternative forced choice task, including integratiion and WM tasks

"""

import neurogym as ngym
from neurogym.ops import tasktools
import numpy as np
from gym import spaces


class GenTask(ngym.EpochEnv):
    metadata = {
        'paper_link': None,
        'paper_name': None,
        'default_timing': {
            'fixation': ('truncated_exponential', [500, 200, 800]),
            'stim1': ('truncated_exponential', [500, 200, 800]),
            'delay_btw_stim': ('truncated_exponential', [500, 200, 800]),
            'stim2': ('truncated_exponential', [500, 200, 800]),
            'delay_aft_stim': ('truncated_exponential', [500, 200, 800]),
            'decision': ('truncated_exponential', [500, 200, 800])},
    }

    def __init__(self, dt=100, timing=None, stimEv=1., noise=0.01,
                 simultaneous_stim=False, cohs=None,
                 gng=False, **kwargs):
        super().__init__(dt=dt, timing=timing)
        self.choices = np.array([1, 2]) - gng
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        if cohs is None:
            cohs = [0, 6.4, 12.8, 25.6, 51.2]
        self.cohs = np.array(cohs)*stimEv
        # Input noise
        self.sigma = np.sqrt(2*100*noise)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)
        self.sim_stim = simultaneous_stim

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        self.firstcounts = True
        self.first_flag = False

        # action and observation spaces
        self.gng = gng*1
        self.action_space = spaces.Discrete(3-self.gng)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """
        self.trial = {
            'ground_truth': self.rng.choice(self.choices),
            'coh': self.rng.choice(self.cohs),
            'sigma_dt': self.sigma_dt,
        }
        self.trial.update(kwargs)
        ground_truth = self.trial['ground_truth']
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        # TODO: Code up the situation of overwriting duration
        # if self.sim_stim:
        #     durs['delay_btw_stim'] = 0

        self.add_epoch('fixation', after=0)
        self.add_epoch('stim1', after='fixation')
        self.add_epoch('delay_btw_stim', after='stim1')
        self.add_epoch('stim2', after='delay_btw_stim')
        self.add_epoch('delay_aft_stim', after='stim2')
        self.add_epoch('decision', after='delay_aft_stim', last_epoch=True)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # TODO: This is converted from previous code, but it doesn't look right
        self.set_ob('fixation', [1, 0, 0])
        high, low = ground_truth+self.gng, 3-(ground_truth+self.gng)

        stim = self.view_ob('stim' + str(high))
        stim[:, 0] = 1
        stim[:, high] = (1 + self.trial['coh']/100)/2
        stim[:, 1:] += np.random.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        stim = self.view_ob('stim' + str(low))
        stim[:, 0] = 1
        stim[:, low] = (1 - self.trial['coh']/100)/2
        stim[:, 1:] += np.random.randn(stim.shape[0], 2) * self.trial['sigma_dt']

        self.set_ob('delay_btw_stim', [1, 0, 0])
        self.set_ob('delay_aft_stim', [1, 0, 0])
        self.set_ob('decision', [0, 0, 0])

        self.set_groundtruth('decision', ground_truth)

        self.first_flag = False

    def _step(self, action, **kwargs):
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
        obs = self.obs_now
        gt = self.gt_now
        # rewards
        reward = 0
        first_trial = np.nan
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                if action == gt:
                    reward = self.R_CORRECT
                    new_trial = True
                else:
                    reward = self.R_FAIL
                    new_trial = self.firstcounts
                if ~self.first_flag:
                    first_trial = 1
                    self.first_flag = True

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt,
                                   'first_trial': first_trial}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.close('all')
#    # GNG
#    timing = {'fixation': [100, 100, 100],
#              'stimulus': [500, 200, 800],
#              'delay_btw_stim': [0, 0, 0],
#              'delay_aft_stim': [200, 100, 300],
#              'decision': [200, 200, 200]}
#    simultaneous_stim = True
#   env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim, gng=True)
#    tasktools.plot_struct(env, def_act=1)
#    plt.title('GNG')
#    # RDM
#    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
#              'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
#    simultaneous_stim = True
#    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
#    tasktools.plot_struct(env)
#    plt.title('RDM')
#    # ROMO
#    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
#              'delay_btw_stim': [500, 200, 800],
#              'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
#    simultaneous_stim = False
#    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
#    tasktools.plot_struct(env)
#    plt.title('ROMO')
#    # DELAY RESPONSE
    timing = get_default_timing()
    timing['fixation'] = ('choice', [200, 300, 400])
    simultaneous_stim = True
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    tasktools.plot_struct(env, num_steps_env=50000, name='DELAY RESPONSE')

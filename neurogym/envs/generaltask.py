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

TIMING = {'fixation': [500, 200, 800], 'stimulus': [500, 200, 800],
          'delay_btw_stim': [500, 200, 800],
          'delay_aft_stim': [500, 200, 800], 'decision': [500, 200, 800]}


class GenTask(ngym.EpochEnv):
    def __init__(self, dt=100, timing=None, stimEv=1., noise=0.01,
                 simultaneous_stim=False, cohs=None,
                 gng=False, **kwargs):
        super().__init__(dt=dt)
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
        if timing is not None:
            for key in timing.keys():
                assert key in TIMING.keys()
                TIMING[key] = timing[key]
        self.timing = TIMING

        self.timing_fn_dict = dict()
        for key, val in TIMING.items():
            self.timing_fn_dict[key] = tasktools.random_number_fn('truncated_exponential', *val)

        self.mean_trial_duration = 0
        self.max_trial_duration = 0
        for key in self.timing.keys():
            self.mean_trial_duration += self.timing[key][0]
            self.max_trial_duration += self.timing[key][2]
            self.timing[key][1] = max(self.timing[key][1], self.dt)
        if not self.sim_stim:
            self.mean_trial_duration += self.timing['stimulus'][0]
            self.max_trial_duration += self.timing['stimulus'][2]

        self.max_steps = int(self.max_trial_duration/dt)

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

    def __str__(self):
        string = ''
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += '2-Alternative Forced Choice Task\n'
        string += 'Mean Fixation: ' + str(self.timing['fixation'][0]) + '\n'
        string += 'Mean stimulus period: ' + str(self.timing['stimulus'][0]) + '\n'
        if not self.sim_stim:
            string += 'Mean delay btw stims: ' + str(self.timing['delay_btw_stim'][0]) + '\n'
        else:
            string += 'stimuli presented simultaneously\n'
        string += 'Mean delay post-stim: ' + str(self.timing['delay_aft_stim'][0]) + '\n'
        string += 'Mean response window: ' + str(self.timing['decision'][0]) + '\n'
        string += 'Mean trial duration : ' + str(self.mean_trial_duration) + '\n'
        string += 'Max trial duration : ' + str(self.max_trial_duration) + '\n'
        string += '(time step: ' + str(self.dt) + '\n'
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        return string

    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of rdm: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
            obs: observation
        """

        if 'gt' in kwargs.keys():
            gt = kwargs['gt']
        else:
            gt = self.rng.choice(self.choices)

        if 'cohs' in kwargs.keys():
            coh = kwargs['cohs']
        else:
            coh = self.rng.choice(self.cohs)

        if 'sigma' in kwargs.keys():
            sigma_dt = kwargs['sigma']
        else:
            sigma_dt = self.sigma_dt

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        if 'durs' not in kwargs.keys():
            durs = dict.fromkeys(TIMING.keys())
            for key in durs.keys():
                # durs[key] = tasktools.trunc_exp(self.rng, self.dt,
                #                                 self.timing[key][0],
                #                                 self.timing[key][1],
                #                                 self.timing[key][2])
                durs[key] = self.timing_fn_dict[key]()
        else:
            durs = kwargs['durs']
            durs_temp = dict.fromkeys(TIMING.keys())
            for key in durs_temp.keys():
                if key in durs.keys():
                    durs_temp[key] = durs[key]
                else:
                    durs[key] = self.timing_fn_dict[key]()

        if self.sim_stim:
            durs['delay_btw_stim'] = 0

        self.add_epoch('fixation', durs['fixation'], after=0)
        self.add_epoch('stim1', durs['stimulus'], after='fixation')
        self.add_epoch('delay_btw_stim', durs['delay_btw_stim'], after='stim1')
        self.add_epoch('stim2', durs['stimulus'], after='delay_btw_stim')
        self.add_epoch('delay_aft_stim', durs['delay_aft_stim'], after='stim2')
        self.add_epoch('decision', durs['decision'], after='delay_aft_stim', last_epoch=True)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # TODO: This is converted from previous code, but it doesn't look right
        self.set_ob('fixation', [1, 0, 0])
        high, low = gt+self.gng, 3-(gt+self.gng)
        tmp = [1, 0, 0]
        tmp[high] = (1 + coh/100)/2
        self.set_ob('stim' + str(high), tmp)
        tmp = [1, 0, 0]
        tmp[low] = (1 - coh/100)/2
        self.set_ob('stim' + str(low), tmp)
        self.set_ob('delay_btw_stim', [1, 0, 0])
        self.set_ob('delay_aft_stim', [1, 0, 0])
        self.set_ob('decision', [0, 0, 0])

        # TODO: Not happy about having to do this ugly thing
        self.obs[self.stim1_ind0:self.stim1_ind1, 1:] += np.random.randn(
            self.stim1_ind1-self.stim1_ind0, 2) * sigma_dt
        self.obs[self.stim2_ind0:self.stim2_ind1, 1:] += np.random.randn(
            self.stim2_ind1 - self.stim2_ind0, 2) * sigma_dt

        self.set_groundtruth('decision', gt)

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
        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]
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
    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
              'delay_aft_stim': [500, 200, 800], 'decision': [100, 100, 100]}
    simultaneous_stim = True
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    tasktools.plot_struct(env, num_steps_env=50000, name='DELAY RESPONSE')

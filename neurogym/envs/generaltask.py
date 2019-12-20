#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:39:17 2019

@author: molano


General two-alternative forced choice task, including integratiion and WM tasks

"""

from neurogym.envs import ngym
from neurogym.ops import tasktools
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
TIMING = {'fixation': [500, 200, 800], 'stimulus': [500, 200, 800],
          'delay_btw_stim': [500, 200, 800],
          'delay_aft_stim': [500, 200, 800], 'decision': [500, 200, 800]}


class GenTask(ngym.ngym):
    def __init__(self, dt=100, timing=None, stimEv=1., noise=0.01,
                 simultaneous_stim=False, cohs=[0, 6.4, 12.8, 25.6, 51.2],
                 gng=False, **kwargs):
        super().__init__(dt=dt)
        self.choices = np.array([1, 2]) - gng
        # cohs specifies the amount of evidence (which is modulated by stimEv)
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
        print('XXXXXXXXXXXXXXXXXXXXXX')
        print('2-Alternative Forced Choice Task')
        print('Mean Fixation: ' + str(self.timing['fixation'][0]))
        print('Mean stimulus period: ' + str(self.timing['stimulus'][0]))
        if not self.sim_stim:
            print('Mean delay btw stims: ' +
                  str(self.timing['delay_btw_stim'][0]))
        else:
            print('stimuli presented simultaneously')
        print('Mean delay post-stim: ' + str(self.timing['delay_aft_stim'][0]))
        print('Mean response window: ' + str(self.timing['decision'][0]))
        print('Mean trial duration : ' + str(self.mean_trial_duration))
        print('Max trial duration : ' + str(self.max_trial_duration))
        print('(time step: ' + str(self.dt) + ')')
        print('XXXXXXXXXXXXXXXXXXXXXX')
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
        # seeding
        self.seed()
        self.viewer = None

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

        if 'gt' in kwargs.keys():
            gt = kwargs['gt']
        else:
            gt = self.rng.choice(self.choices)
        self.gt = gt

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
                durs[key] = tasktools.trunc_exp(self.rng, self.dt,
                                                self.timing[key][0],
                                                self.timing[key][1],
                                                self.timing[key][2])
        else:
            durs = kwargs['durs']
            durs_temp = dict.fromkeys(TIMING.keys())
            for key in durs_temp.keys():
                if key in durs.keys():
                    durs_temp[key] = durs[key]
                else:
                    durs[key] = tasktools.trunc_exp(self.rng, self.dt,
                                                    self.timing[key][0],
                                                    self.timing[key][1],
                                                    self.timing[key][2])
        if self.sim_stim:
            durs['delay_btw_stim'] = 0
        # trial duration
        self.tmax = np.sum([durs[key] for key in durs.keys()])
        if not self.sim_stim:
            self.tmax += durs['stimulus']
        self.pers = {}
        per_times = {}
        periods = ['fixation', 'stim_1', 'delay_btw_stim', 'stim_2',
                   'delay_aft_stim', 'decision']
        t = np.arange(0, self.tmax, self.dt)
        cum = 0
        for key in periods:
            cum_aux = 0
            if key != 'stim_1' and key != 'stim_2':
                cum_aux = durs[key]
            if key == 'fixation':
                self.pers[key] = [cum, cum + durs[key]]
            elif key == 'stim_1':
                per_times[key] = np.logical_and(t >= cum,
                                                t < cum + durs['stimulus'])
                if not self.sim_stim:
                    cum_aux = durs['stimulus']
            elif key == 'stim_2':
                cum_aux = durs['stimulus']
                per_times[key] = np.logical_and(t >= cum, t < cum + cum_aux)
            elif key == 'decision':
                self.pers[key] = [cum, cum + durs[key]]
                per_times[key] = np.logical_and(t >= cum, t < cum + cum_aux)
            cum += cum_aux

        n_stim = int(durs['stimulus']/self.dt)

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # observations
        obs = np.zeros((len(t), 3))
        # fixation cue is always on except in decision period
        obs[~per_times['decision'], 0] = 1
        # correct stimulus
        obs[per_times['stim_' + str((gt+self.gng))],
            (gt+self.gng)] = (1 + coh/100)/2
        obs[per_times['stim_' + str((gt+self.gng))],
            (gt+self.gng)] += np.random.randn(n_stim) * sigma_dt
        # incorrect stimulus
        obs[per_times['stim_' + str(3-(gt+self.gng))],
            3-(gt+self.gng)] = (1 - coh/100)/2
        obs[per_times['stim_' + str(3-(gt+self.gng))],
            3-(gt+self.gng)] += np.random.randn(n_stim) * sigma_dt

        self.obs = obs

        self.t = 0
        self.num_tr += 1
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
        if self.num_tr == 0:
            # start first trial
            self.new_trial()
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = np.zeros((3-self.gng,))
        first_trial = np.nan
        if self.pers['fixation'][0] <= self.t < self.pers['fixation'][1]:
            gt[0] = 1
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.pers['decision'][0] <= self.t < self.pers['decision'][1]:
            gt[self.gt] = 1
            if action != 0:
                if action == self.gt:
                    reward = self.R_CORRECT
                    new_trial = True
                else:
                    reward = self.R_FAIL
                    new_trial = self.firstcounts
                if ~self.first_flag:
                    first_trial = 1
                    self.first_flag = True
        else:
            gt[0] = 1
        obs = self.obs[int(self.t/self.dt), :]

        # ---------------------------------------------------------------------
        # new trial?
        reward, new_trial = tasktools.new_trial(self.t, self.tmax,
                                                self.dt, new_trial,
                                                self.R_MISS, reward)
        self.t += self.dt

        done = self.num_tr > self.num_tr_exp

        return obs, reward, done, {'new_trial': new_trial, 'gt': gt,
                                   'first_trial': first_trial}

    def step(self, action):
        """
        step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        Note that the main computations are done by the function _step(action),
        and the extra lines are basically checking whether to call the
        new_trial() function in order to start a new trial
        """
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.new_trial()
        return obs, reward, done, info


if __name__ == '__main__':
    plt.close('all')
    # GNG
    timing = {'fixation': [100, 100, 100],
              'stimulus': [500, 200, 800],
              'delay_btw_stim': [0, 0, 0],
              'delay_aft_stim': [200, 100, 300],
              'decision': [200, 200, 200]}
    simultaneous_stim = True
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim, gng=True)
    tasktools.plot_struct(env, def_act=1)
    plt.title('GNG')
    # RDM
    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
              'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
    simultaneous_stim = True
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    tasktools.plot_struct(env)
    plt.title('RDM')
    # ROMO
    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
              'delay_btw_stim': [500, 200, 800],
              'delay_aft_stim': [0, 0, 0], 'decision': [100, 100, 100]}
    simultaneous_stim = False
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    tasktools.plot_struct(env)
    plt.title('ROMO')
    # DELAY RESPONSE
    timing = {'fixation': [500, 500, 500], 'stimulus': [500, 200, 800],
              'delay_aft_stim': [500, 200, 800], 'decision': [100, 100, 100]}
    simultaneous_stim = True
    env = GenTask(timing=timing, simultaneous_stim=simultaneous_stim)
    tasktools.plot_struct(env)
    plt.title('DELAY RESPONSE')

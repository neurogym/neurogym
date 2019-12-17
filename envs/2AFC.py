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


class TwoAFC(ngym.ngym):
    def __init__(self, dt=100, timing=None, stimEv=1., noise=0.01,
                 simultaneous_stim=False, **kwargs):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
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
        # action and observation spaces
        self.action_space = spaces.Discrete(3)
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
            ground_truth = kwargs['gt']
        else:
            ground_truth = self.rng.choice(self.choices)
        self.ground_truth = ground_truth

        if 'coh' in kwargs.keys():
            coh = kwargs['coh']
        else:
            coh = self.rng.choice(self.cohs)
        self.coh = coh

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
        if self.sim_stim:
            durs['delay_btw_stim'] = 0
        # trial duration
        self.tmax = np.sum([durs[key] for key in durs.keys()])
        self.pers = {}
        per_times = {'fixation': None, 'stim_1': None, 'stim_2': None,
                     'delay_btw_stim': None, 'delay_aft_stim': None,
                     'decision': None}
        t = np.arange(0, self.tmax, self.dt)
        cum = 0
        plt.figure()
        counter = 0
        for key in per_times.keys():
            counter += 1
            print(cum)
            if key == 'decision' or key == 'fixation':
                self.pers[key] = [cum, cum + durs[key]]
            if key == 'stim_1':
                per_times[key] =\
                    np.logical_and(t >= cum, t < cum + durs['stimulus'])
                if not self.sim_stim:
                    cum += durs['stimulus']
            elif key == 'stim_2':
                per_times[key] =\
                    np.logical_and(t >= cum, t < cum + durs['stimulus'])
                cum += durs['stimulus']
            else:
                per_times[key] =\
                    np.logical_and(t >= cum, t < cum + durs[key])
                cum += durs[key]
            plt.plot(per_times[key]+counter, label=key, lw=2)
        n_stim = int(durs['stimulus']/self.dt)
        plt.legend()
        
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        # observations
        obs = np.zeros((len(t), 3))
        # fixation cue is always on except in decision period
        obs[per_times['fixation'], 0] = 1
        obs[per_times['stim_1'], 0] = 1
        obs[per_times['delay_btw_stim'], 0] = 1
        obs[per_times['stim_2'], 0] = 1
        obs[per_times['delay_aft_stim'], 0] = 1
        # correct stimulus
        obs[per_times['stim_' + str(ground_truth)],
            ground_truth] = (1 + coh/100)/2
        obs[per_times['stim_' + str(ground_truth)],
            ground_truth] += np.random.randn(n_stim) * self.sigma_dt
        # incorrect stimulus
        obs[per_times['stim_' + str(3 - ground_truth)],
            3 - ground_truth] = (1 - coh/100)/2
        obs[per_times['stim_' + str(3 - ground_truth)],
            3 - ground_truth] += np.random.randn(n_stim) * self.sigma_dt
        self.obs = obs
        # ground truth
        self.gt = np.zeros((len(t),), dtype=np.int)
        self.gt[per_times['decision']] = self.ground_truth

        self.t = 0
        self.num_tr += 1

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
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = np.zeros((3,))
        if self.pers['fixation'][0] <= self.t < self.pers['fixation'][1]:
            gt[0] = 1
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.pers['decision'][0] <= self.t < self.pers['decision'][1]:
            gt[self.ground_truth] = 1
            if self.ground_truth == action:
                reward = self.R_CORRECT
            elif self.ground_truth != 0:  # 3-action is the other act
                reward = self.R_FAIL
            new_trial = action != 0
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

        return obs, reward, done, {'new_trial': new_trial, 'gt': gt}

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
        if self.num_tr == 0:
            # start first trial
            self.new_trial()

        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.new_trial()
        return obs, reward, done, info


if __name__ == '__main__':
    # RDM
    timing = {'fixation': [500, 200, 800], 'stimulus': [500, 200, 800],
              'delay_btw_stim': [0, 0, 0],
              'delay_aft_stim': [0, 0, 0], 'decision': [500, 200, 800]}

    env = TwoAFC(timing=timing)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    num_steps_env = 100
    for stp in range(int(num_steps_env)):
        action = 1  # env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        observations.append(obs)
        if info['new_trial']:
            actions_end_of_trial.append(action)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
        if 'config' in info.keys():
            config_mat.append(info['config'])
        else:
            config_mat.append([0, 0])

    rows = 3
    obs = np.array(observations)
    plt.figure()
    plt.subplot(rows, 1, 1)
    plt.imshow(obs.T, aspect='auto')
    plt.title('observations')
    plt.subplot(rows, 1, 2)
    plt.plot(actions, marker='+')
    #    plt.plot(actions_end_of_trial, '--')
    gt = np.array(gt)
    plt.plot(np.argmax(gt, axis=1), 'r')
    #    # aux = np.argmax(obs, axis=1)
    # aux[np.sum(obs, axis=1) == 0] = -1
    # plt.plot(aux, '--k')
    plt.title('actions')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.subplot(rows, 1, 3)
    plt.plot(rewards, 'r')
    plt.title('reward')
    plt.xlim([-0.5, len(rewards)+0.5])
    plt.show()

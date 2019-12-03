#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:58:10 2019

@author: molano
"""
from neurogym.envs import ngym
from neurogym.ops import tasktools
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


class DR(ngym.ngym):
    def __init__(self, dt=100, timing=(500, 80, 330, 1500, 500),
                 stimEv=1., delays=[1000, 5000, 10000], **kwargs):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2])*stimEv
        # Input noise
        self.sigma = np.sqrt(2*100*0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)
        # TODO: this is not natural
        self.fixation = timing[0]
        self.stimulus_min = timing[1]
        self.stimulus_mean = timing[2]
        self.stimulus_max = timing[3]
        self.decision = timing[4]
        self.delays = delays
        # TODO: How to make this easier?
        self.max_trial_duration = self.fixation + self.stimulus_max +\
            np.max(self.delays) + self.decision
        self.max_steps = int(self.max_trial_duration/dt)
        if (self.fixation == 0 or self.decision == 0 or
           self.stimulus_mean == 0):
            print('XXXXXXXXXXXXXXXXXXXXXX')
            print('the duration of all periods must be larger than 0')
            print('XXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXX')
        print('Random Dots Motion Task')
        print('Mean Fixation: ' + str(self.fixation))
        print('Min Stimulus Duration: ' + str(self.stimulus_min))
        print('Mean Stimulus Duration: ' + str(self.stimulus_mean))
        print('Max Stimulus Duration: ' + str(self.stimulus_max))
        print('Delay: ' + str(self.delays))
        print('Decision: ' + str(self.decision))
        print('(time step: ' + str(self.dt) + ')')
        print('Mean Trial Duration: ' + str(self.max_trial_duration))
        print('XXXXXXXXXXXXXXXXXXXXXX')
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -1.
        self.R_MISS = 0.
        self.abort = False
        self.firstcounts = True
        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

        # start new trial
        # self.trial = self.new_trial()
        self.new_trial()

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
        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        if 'durs' in kwargs.keys():
            fixation = kwargs['durs'][0]
            stimulus = kwargs['durs'][1]
            delay = kwargs['durs'][2]
            decision = kwargs['durs'][3]
        else:
            stimulus = tasktools.truncated_exponential(self.rng, self.dt,
                                                       self.stimulus_mean,
                                                       xmin=self.stimulus_min,
                                                       xmax=self.stimulus_max)
            delay = self.rng.choice(self.delays)
            fixation = self.fixation
            decision = self.decision

        # maximum length of current trial
        self.tmax = fixation + stimulus + delay + decision
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice(self.choices)
        coh = self.rng.choice(self.cohs)
        self.ground_truth = ground_truth
        self.coh = coh
        t = np.arange(0, self.tmax, self.dt)

        # define periods
        self.fixation_0 = 0
        self.fixation_1 = fixation
        self.stimulus_0 = fixation
        self.stimulus_1 = fixation + stimulus
        self.delay_0 = fixation + stimulus
        self.delay_1 = fixation + stimulus + delay
        self.decision_0 = fixation + stimulus + delay
        self.decision_1 = fixation + stimulus + delay + decision
        fixation_period = np.logical_and(t >= self.fixation_0,
                                         t < self.fixation_1)
        stimulus_period = np.logical_and(t >= self.stimulus_0,
                                         t < self.stimulus_1)
        delay_period = np.logical_and(t >= self.delay_0,
                                      t < self.delay_1)
        decision_period = np.logical_and(t >= self.decision_0,
                                         t < self.decision_1)
        # define observations
        obs = np.zeros((len(t), 3))
        obs[fixation_period, 0] = 1
        obs[delay_period, 0] = 1
        n_stim = int(stimulus/self.dt)
        obs[stimulus_period, 0] = 1
        obs[stimulus_period, ground_truth] = (1 + coh/100)/2
        obs[stimulus_period, 3 - ground_truth] = (1 - coh/100)/2
        obs[stimulus_period, 1:] += np.random.randn(n_stim, 2) * self.sigma_dt
        self.obs = obs
        self.t = 0
        self.num_tr += 1
        self.gt = np.zeros((len(t),), dtype=np.int)
        self.gt[decision_period] = self.ground_truth

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
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        # observations
        gt = np.zeros((3,))
        if ((self.fixation_0 <= self.t < self.fixation_1) or
           (self.delay_0 <= self.t < self.delay_1)):
            gt[0] = 1
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.decision_0 <= self.t < self.decision_1:
            gt[self.ground_truth] = 1
            if self.ground_truth == action:
                reward = self.R_CORRECT
                new_trial = True
            elif self.ground_truth == 3 - action:  # 3-action is the other act
                reward = self.R_FAIL
                new_trial = self.firstcounts
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
        obs, reward, done, info = self._step(action)
        if info['new_trial']:
            self.new_trial()
        return obs, reward, done, info


if __name__ == '__main__':
    env = DR(timing=[100, 200, 200, 200, 100])
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

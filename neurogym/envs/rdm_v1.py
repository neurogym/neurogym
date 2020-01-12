#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:55:36 2019

@author: molano
"""

import numpy as np
from gym import spaces

import neurogym as ngym
from neurogym.ops import tasktools


class RDM(ngym.EpochEnv):
    def __init__(self, dt=100, timing=(500, 80, 330, 1500, 500),
                 stimEv=1., **kwargs):
        super().__init__(dt=dt)
        self.choices = [1, 2]
        # cohs specifies the amount of evidence (which is modulated by stimEv)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stimEv
        # Input noise
        self.sigma = np.sqrt(2 * 100 * 0.01)
        self.sigma_dt = self.sigma / np.sqrt(self.dt)
        # Durations (stimulus duration will be drawn from an exponential)
        # TODO: this is not natural
        self.fixation = timing[0]
        self.stimulus_min = timing[1]
        self.stimulus_mean = timing[2]
        self.stimulus_max = timing[3]
        self.decision = timing[4]
        self.mean_trial_duration = self.fixation + self.stimulus_mean + \
                                   self.decision
        # TODO: How to make this easier?
        self.max_trial_duration = self.fixation + self.stimulus_max + \
                                  self.decision
        self.max_steps = int(self.max_trial_duration / dt)

        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.R_MISS = 0.
        self.abort = False
        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)
        # seeding
        self.seed()
        self.viewer = None

    def __str__(self):
        string = ''
        if (self.fixation == 0 or self.decision == 0 or
                self.stimulus_mean == 0):
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'Random Dots Motion Task\n'
        string += 'Mean Fixation: ' + str(self.fixation) + '\n'
        string += 'Min Stimulus Duration: ' + str(self.stimulus_min) + '\n'
        string += 'Mean Stimulus Duration: ' + str(self.stimulus_mean) + '\n'
        string += 'Max Stimulus Duration: ' + str(self.stimulus_max) + '\n'
        string += 'Decision: ' + str(self.decision) + '\n'
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
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        ground_truth = self.rng.choice(self.choices)
        coh = self.rng.choice(self.cohs)
        self.ground_truth = ground_truth
        self.coh = coh

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        stimulus = tasktools.trunc_exp(self.rng, self.dt,
                                       self.stimulus_mean,
                                       xmin=self.stimulus_min,
                                       xmax=self.stimulus_max)
        # fixation = self.rng.uniform(self.fixation_min, self.fixation_max)
        fixation = self.fixation
        decision = self.decision

        # maximum length of current trial
        self.add_epoch('fixation', duration=fixation, start=0)
        self.add_epoch('stimulus', duration=stimulus, after='fixation')
        self.add_epoch('decision', duration=decision, after='stimulus', last_epoch=True)

        self.set_ob('fixation', [1, 0, 0])
        if ground_truth == 1:
            self.set_ob('stimulus', [1,  (1 + coh / 100) / 2, (1 - coh / 100) / 2])
        else:
            self.set_ob('stimulus', [1,  (1 - coh / 100) / 2, (1 + coh / 100) / 2])

        self.obs[self.stimulus_ind0:self.stimulus_ind1] += np.random.randn(
            *self.obs[self.stimulus_ind0:self.stimulus_ind1].shape) * self.sigma_dt

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
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        obs = self.obs[self.t_ind]
        gt = self.gt[self.t_ind]
        # observations
        if self.in_epoch('fixation'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt}



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = RDM(timing=[100, 200, 200, 200, 100])
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

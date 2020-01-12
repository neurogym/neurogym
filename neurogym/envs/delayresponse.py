#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:58:10 2019

@author: molano
"""
import neurogym as ngym
from neurogym.ops import tasktools
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt


class DR(ngym.EpochEnv):
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
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = -1.
        self.R_MISS = 0.
        self.abort = False
        self.firstcounts = True
        self.first_flag = False
        # action and observation spaces
        self.stimulus_min = np.max([self.stimulus_min, dt])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def __str__(self):
        string = ''
        if (self.fixation == 0 or self.decision == 0 or
           self.stimulus_mean == 0):
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
            string += 'the duration of all periods must be larger than 0\n'
            string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'XXXXXXXXXXXXXXXXXXXXXX\n'
        string += 'Delay Response Task\n'
        string += 'Mean Fixation: ' + str(self.fixation) + '\n'
        string += 'Min Stimulus Duration: ' + str(self.stimulus_min) + '\n'
        string += 'Mean Stimulus Duration: ' + str(self.stimulus_mean) + '\n'
        string += 'Max Stimulus Duration: ' + str(self.stimulus_max) + '\n'
        string += 'Delay: ' + str(self.delays) + '\n'
        string += 'Decision: ' + str(self.decision) + '\n'
        string += '(time step: ' + str(self.dt) + ')\n'
        string += 'Mean Trial Duration: ' + str(self.max_trial_duration) + '\n'
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
        self.first_flag = False

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        if 'gt' in kwargs.keys():
            ground_truth = kwargs['gt']
        else:
            ground_truth = self.rng.choice(self.choices)
        if 'cohs' in kwargs.keys():
            coh = self.rng.choice(kwargs['cohs'])
        else:
            coh = self.rng.choice(self.cohs)
        if 'sigma' in kwargs.keys():
            sigma = kwargs['sigma'] / np.sqrt(self.dt)
        else:
            sigma = self.sigma_dt

        # ---------------------------------------------------------------------
        # Epochs
        # ---------------------------------------------------------------------
        if 'durs' in kwargs.keys():
            fixation = kwargs['durs'][0]
            stimulus = kwargs['durs'][1]
            delay = kwargs['durs'][2]
            decision = kwargs['durs'][3]
        else:
            stimulus = tasktools.trunc_exp(self.rng, self.dt,
                                           self.stimulus_mean,
                                           xmin=self.stimulus_min,
                                           xmax=self.stimulus_max)
            delay = self.rng.choice(self.delays)
            fixation = self.fixation
            decision = self.decision


        self.ground_truth = ground_truth
        self.coh = coh

        self.add_epoch('fixation', fixation, start=0)
        self.add_epoch('stimulus', stimulus, after='fixation')
        self.add_epoch('delay', delay, after='stimulus')
        self.add_epoch('decision', decision, after='delay', last_epoch=True)

        # define observations
        self.set_ob('fixation', [1, 0, 0])
        stimulus_value = [1] + [(1 - coh/100)/2] * 2
        stimulus_value[ground_truth] = (1 + coh/100)/2
        self.set_ob('stimulus', stimulus_value)
        self.set_ob('delay', [1, 0, 0])
        self.obs[self.stimulus_ind0:self.stimulus_ind1, 1:] += np.random.randn(
            self.stimulus_ind1-self.stimulus_ind0, 2) * sigma

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
        # observations
        obs = self.obs[self.t_ind, :]
        gt = self.gt[self.t_ind]

        first_trial = np.nan
        if self.in_epoch('fixation') or self.in_epoch('delay'):
            if action != 0:
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_epoch('decision'):
            if self.ground_truth == action:
                reward = self.R_CORRECT
                new_trial = True
                if ~self.first_flag:
                    first_trial = True
                    self.first_flag = True
            elif self.ground_truth == 3 - action:  # 3-action is the other act
                reward = self.R_FAIL
                new_trial = self.firstcounts
                if ~self.first_flag:
                    first_trial = False
                    self.first_flag = True

        return obs, reward, False, {'new_trial': new_trial, 'gt': gt,
                                   'first_trial': first_trial}


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
        action = 2  # env.action_space.sample()
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

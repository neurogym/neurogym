#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:25:12 2019

@author: linux

@author: molano
"""

import numpy as np
from gym.core import Wrapper
from gym import spaces


class PassReward(Wrapper):
    metadata = {
        'description': """Modifies observation by adding the previous
        reward.""",
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        super().__init__(env)
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([reward])))
        return obs, reward, done, info


class MissTrialReward(Wrapper):
    metadata = {
        'description': """Add a negative reward if a trial ends with no action.""",
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env, r_miss=0.):
        super().__init__(env)
        self.r_miss = r_miss

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['trial_endwith_tmax']:
            reward += self.r_miss
        return obs, reward, done, info



if __name__ == '__main__':
    from neurogym.envs import nalt_rdm
    from neurogym.wrappers import trial_hist_nalt
    from neurogym.wrappers import catch_trials
    import matplotlib.pyplot as plt

    env = nalt_rdm.nalt_RDM(timing=[100, 200, 200, 200, 100], n=10)
    env = trial_hist_nalt.TrialHistory_NAlt(env)
    env = catch_trials.CatchTrials(env, catch_prob=0.7, stim_th=100)
    env = PassReward(env)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    num_steps_env = 1000
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
    print(np.sum(np.array(rewards) == 1) / np.sum(np.argmax(gt, axis=1) == 1))

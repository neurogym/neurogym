#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:53:05 2019

@author: linux

@author: molano
"""

import numpy as np
from gym.core import Wrapper
from gym import spaces


class PassAction(Wrapper):
    metadata = {
        'description': 'Modifies observation by adding the previous action.',
        'paper_link': None,
        'paper_name': None,
    }

    def __init__(self, env):
        """
        Modifies observation by adding the previous action.
        """
        super().__init__(env)
        self.env = env
        # TODO: This is not adding one-hot
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(env_oss+1,),
                                            dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return np.concatenate((obs, np.array([0])))

    def step(self, action):
        # TODO: Need to turn action into one-hot
        obs, reward, done, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, done, info


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from neurogym.envs import nalt_perceptualDecisionMaking
    from neurogym.wrappers import trial_hist_nalt
    from neurogym.wrappers import catch_trials
    from neurogym.wrappers import pass_reward
    n_ch = 3
    env = nalt_perceptualDecisionMaking.nalt_PerceptualDecisionMaking(timing=[100, 200, 200, 200, 100], n_ch=n_ch)
    env = trial_hist_nalt.TrialHistory_NAlt(env, n_ch=n_ch, tr_prob=0.9,
                                            trans='RepAlt')
    env = catch_trials.CatchTrials(env, catch_prob=0.7, stim_th=100)
    env = pass_reward.PassReward(env)
    env = PassAction(env)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    num_steps_env = 200
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

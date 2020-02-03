#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:45:32 2019

@author: molano
"""
from gym.core import Wrapper
import numpy as np


class Noise(Wrapper):
    metadata = {
        'description': 'Add Gaussian noise to the observations.',
        'paper_link': None,
        'paper_name': None,
        'std_noise': 'Standard deviation of noise. (def: 0.1)',
        'w': 'Window length. (def: 100)'
    }

    def __init__(self, env, std_noise=.1, rew_th=None, w=100):
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise
        self.std_noise = self.std_noise / self.env.dt
        self.init_noise = self.std_noise
        self.rewards = []
        self.w = w
        self.min_w = False
        self.add_noise = False
        if rew_th is not None:
            self.rew_th = rew_th
        else:
            self.rew_th = 0
            self.min_w = True
            self.init_noise = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if info['new_trial']:
            self.rewards.append(reward)
            if len(self.rewards) > self.w:
                self.rewards.pop(0)
                self.min_w = True

        rew_mean = np.mean(self.rewards)
        info['rew_mean'] = rew_mean
        info['std_noise'] = self.std_noise * self.min_w

        if rew_mean >= self.rew_th and self.min_w is True:
            self.add_noise = True
            obs += np.random.normal(loc=0, scale=self.std_noise,
                                    size=obs.shape)
            self.std_noise += self.init_noise
        elif self.add_noise is True:
            obs += np.random.normal(loc=0, scale=self.std_noise,
                                    size=obs.shape)

        return obs, reward, done, info


def plot_env(env, num_steps_env=200,
             def_act=None, model=None, name=None, legend=True):
    # TODO: Separate the running from plotting. Make running a separate function
    # TODO: Can't we use Monitor here?
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations = []
    obs_cum = []
    state_mat = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    perf = []
    std_noise = []
    rew_mean = []
    obs = env.reset()
    obs_cum_temp = obs
    for stp in range(int(num_steps_env)):
        if model is not None:
            # TODO: This is a particular kind of model. Document.
            action, _states = model.predict(obs)
            if isinstance(action, float) or isinstance(action, int):
                action = [action]
            state_mat.append(_states)
        elif def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        obs_cum_temp += obs
        obs_cum.append(obs_cum_temp.copy())
        if isinstance(info, list):
            info = info[0]
            obs_aux = obs[0]
            rew = rew[0]
            done = done[0]
            action = action[0]
        else:
            obs_aux = obs

        if done:
            env.reset()
        observations.append(obs_aux)
        if info['new_trial']:
            actions_end_of_trial.append(action)
            perf.append(rew)
            obs_cum_temp = np.zeros_like(obs_cum_temp)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
        std_noise.append(info['std_noise'])
        rew_mean.append(info['rew_mean'])
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    obs_cum = np.array(obs_cum)
    obs = np.array(observations)
    fig_(obs, actions, gt, rewards, std_noise, rew_mean, legend=legend,
         name=name)
    data = {'obs': obs, 'obs_cum': obs_cum, 'rewards': rewards,
            'actions': actions, 'perf': perf,
            'actions_end_of_trial': actions_end_of_trial, 'gt': gt,
            'states': states}
    return data


def fig_(obs, actions, gt=None, rewards=None, std_noise=None, rew_mean=None,
         legend=True, name='', folder=''):
    if len(obs.shape) != 2:
        raise ValueError('obs has to be 2-dimensional.')
    # TODO: Add documentation
    steps = np.arange(obs.shape[0])

    n_row = 2  # observation and action
    n_row += rewards is not None
    n_row += std_noise is not None
    n_row += rew_mean is not None

    gt_colors = 'gkmcry'
    f, axes = plt.subplots(n_row, 1, sharex=True, figsize=(5, n_row*1.5))
    # obs
    ax = axes[0]
    ax.imshow(obs.T, aspect='auto')
    if name:
        ax.set_title(name + ' env')
    ax.set_ylabel('Observations')
    ax.set_yticks([])
    ax.set_xlim([-0.5, len(steps)-0.5])

    # actions
    ax = axes[1]
    ax.plot(steps, actions, marker='+', label='Actions')

    if gt is not None:
        gt = np.array(gt)
        if len(gt.shape) > 1:
            for ind_gt in range(gt.shape[1]):
                ax.plot(steps, gt[:, ind_gt], '--'+gt_colors[ind_gt],
                        label='Ground truth '+str(ind_gt))
        else:
            ax.plot(steps, gt, '--'+gt_colors[0], label='Ground truth')

    ax.set_ylabel('Actions')
    if legend:
        ax.legend()

    if rewards is not None:
        # rewards
        ax = axes[2]
        ax.plot(steps, rewards, 'r')
        ax.set_ylabel('Reward')
        # ax.set_ylabel('reward ' + ' (' + str(np.round(np.mean(perf), 2)) + ')')

    if std_noise is not None:
        ax.set_xticks([])
        ax = axes[3]
        ax.plot(steps, std_noise)
        ax.set_title('Noise')
        ax.set_ylabel('Std')

    if rew_mean is not None:
        ax.set_xticks([])
        ax = axes[4]
        ax.plot(steps, rew_mean)
        ax.set_title('Rew window')
        ax.set_ylabel('mean rew')

    ax.set_xlabel('Steps')
    plt.tight_layout()
    plt.show()
    if folder != '':
        f.savefig(folder + '/env_struct.png')
        plt.close(f)

    return f

import gym
import matplotlib.pyplot as plt
if __name__ == '__main__':
    task = 'PerceptualDecisionMakingDelayResponse-v0'
    env = gym.make(task)
    env = Noise(env, rew_th=0.1)
    plot_env(env, num_steps_env=20000)

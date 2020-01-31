#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:10:01 2020

@author: martafradera
"""

from gym.core import Wrapper
import neurogym
import gym
import matplotlib.pyplot as plt
import numpy as np


class TTLPulse(Wrapper):
    metadata = {
        'description': 'Outputs extra pulses that will be non-zero during ' +
        'specified periods.',
        'paper_link': None,
        'paper_name': None,
        'periods': 'List of list specifying the on periods for each pulse. ' +
        '(def: [])'
    }

    def __init__(self, env, periods=[]):
        super().__init__(env)

        self.periods = periods

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for ind_p, periods in enumerate(self.periods):
            info['signal_' + str(ind_p)] = 0
            for per in periods:
                if self.env.in_period(per):
                    info['signal_' + str(ind_p)] = 1

        return obs, reward, done, info


def plot_struct(env, num_steps_env=200, n_stps_plt=200,
                def_act=None, model=None, name=None, legend=True):
    if isinstance(env, str):
        env = gym.make(env)
    if name is None:
        name = type(env).__name__
    observations = []
    state_mat = []
    rewards = []
    actions = []
    signal1 = []
    signal2 = []

    obs = env.reset()
    for stp in range(int(num_steps_env)):
        if def_act is not None:
            action = def_act
        else:
            action = env.action_space.sample()
        obs, rew, done, info = env.step(action)

        if done:
            env.reset()
        observations.append(obs)
        rewards.append(rew)
        actions.append(action)
        signal1.append(info['signal_0'])
        signal2.append(info['signal_1'])
    if model is not None:
        states = np.array(state_mat)
        states = states[:, 0, :]
    else:
        states = None
    obs = np.array(observations)
    fig_(obs, actions, rewards, signal1, signal2, n_stps_plt)


def fig_(obs, actions, rewards, signal1, signal2, n_stps_plt):
    rows = 4

    plt.figure(figsize=(8, 8))
    # obs
    plt.subplot(rows, 1, 1)
    plt.imshow(obs[:n_stps_plt, :].T, aspect='auto')
    plt.title('observations')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    # actions
    plt.subplot(rows, 1, 2)
    plt.plot(np.arange(n_stps_plt) + 0.,
             actions[:n_stps_plt], marker='+', label='actions')
    plt.ylabel('actions')
    plt.legend()
    plt.xlim([-0.5, n_stps_plt-0.5])
    ax = plt.gca()
    ax.set_xticks([])
    # rewards
    plt.subplot(rows, 1, 3)
    plt.plot(np.arange(n_stps_plt) + 0.,
             rewards[:n_stps_plt], 'r')
    plt.xlim([-0.5, n_stps_plt-0.5])
    plt.ylabel('reward ')
    plt.subplot(rows, 1, 4)
    plt.plot(np.arange(n_stps_plt) + 0.,
             signal1[:n_stps_plt], 'r')
    plt.plot(np.arange(n_stps_plt) + 0.,
             signal2[:n_stps_plt], 'b')
    plt.xlim([-0.5, n_stps_plt-0.5])
    plt.ylabel('reward ')

    plt.xlabel('timesteps')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    task = 'GoNogo-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stimulus': ('constant', 500),
                                    'resp_delay': ('constant', 500),
                                    'decision': ('constant', 500)}}
    env = gym.make(task, **KWARGS)
    env = TTLPulse(env, periods=[['resp_delay'], ['decision']])
    plot_struct(env, num_steps_env=100, n_stps_plt=100)

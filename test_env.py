#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:46:43 2019

@author: gryang
"""
import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym


def test_run(env_name):
    """Test if all one environment can at least be run."""
    kwargs = {'dt': 100}
    env = gym.make(env_name, **kwargs)
    env.reset()
    for stp in range(100):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()
    return env


def test_run_all():
    """Test if all environments can at least be run."""
    from neurogym import all_tasks
    success_count = 0
    total_count = 0
    for env_name in sorted(all_tasks.keys()):
        total_count += 1
        try:
            env = test_run(env_name)
            print('Success at running env: {:s}'.format(env_name))
            # print(env)
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} tasks'.format(success_count, total_count))


def test_plot(env_name):
    kwargs = {'dt': 100}
    env = gym.make(env_name, **kwargs)

    env.reset()
    observations = []
    for stp in range(100):
        if np.mod(stp, 2) == 0:
            action = 0
        else:
            action = 0
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        observations.append(state)

        # print(state)
        # print(info)
        # print(rew)
        # print(info)
    obs = np.array(observations)
    plt.figure()
    plt.imshow(obs.T, aspect='auto')
    plt.show()


if __name__ == '__main__':
    # test_run_all()
    test_plot('DelayedMatchCategory-v0')
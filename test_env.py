#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test environments"""

import time
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


def test_speed(env_name):
    """Test if all one environment can at least be run."""
    n_steps = 100000
    kwargs = {'dt': 100}

    total_time = 0
    env = gym.make(env_name, **kwargs)
    env.reset()
    for stp in range(n_steps):
        action = env.action_space.sample()
        start_time = time.time()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        total_time += time.time() - start_time
        if done:
            env.reset()

    print('Time per step {:0.3f}us'.format(total_time/n_steps*1e6))
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
    kwargs = {'dt': 13}
    env = gym.make(env_name, **kwargs)

    env.reset()
    observations = []
    for stp in range(200):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        print(obs.shape)
        observations.append(obs)

        # print(state)
        # print(info)
        # print(rew)
        # print(info)
    observations = np.array(observations)
    plt.figure()
    plt.imshow(observations.T, aspect='auto')
    plt.show()


if __name__ == '__main__':
    # test_run_all()
    # env_name = 'RDM-v1'
    # env_name = 'DelayedMatchCategory-v0'
    env_name = 'MemoryRecall-v0'
    test_plot(env_name)
    # test_speed(env_name)
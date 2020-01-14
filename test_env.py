#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test environments"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym
from neurogym import all_tasks


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
    """Test speed of an environment."""
    n_steps = 100000
    warmup_steps = 10000
    kwargs = {'dt': 20}

    env = gym.make(env_name, **kwargs)
    env.reset()
    for stp in range(warmup_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

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

    print('Env: ', env_name)
    print('Time per step {:0.3f}us'.format(total_time/n_steps*1e6))
    return env


def test_speed_all():
    """Test speed of all experiments."""
    for env_name in sorted(all_tasks.keys()):
        try:
            test_speed(env_name)
            print('Success')
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)


def test_print_all():
    """Test printing of all experiments."""
    for env_name in sorted(all_tasks.keys()):
        print('')
        print('Test printing env: {:s}'.format(env_name))
        try:
            env = gym.make(env_name)
            print(env)
            print('Success')
        except BaseException as e:
            print('Failure')
            print(e)


def test_run_all():
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    for env_name in sorted(all_tasks.keys()):
        if env_name in ['Combine-v0']:
            continue
        total_count += 1

        print('Running env: {:s}'.format(env_name))
        # env = test_run(env_name)
        try:
            test_run(env_name)
            print('Success')
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
    for stp in range(500):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
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
    test_run_all()
    # test_speed_all()
    # test_print_all()
    # env_name = 'GenTask-v0'
    # env_name = 'RDM-v1'
    # env_name = 'DPA-v1'
    # env_name = 'NAltRDM-v0'
    # env_name = 'DelayedMatchCategory-v0'
    # env_name = 'MemoryRecall-v0'
    env_name = 'Bandit-v0'
    test_plot(env_name)
    # test_speed(env_name)

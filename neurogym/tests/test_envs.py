#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test environments.

All tests in this file can be run by running in command line
py.test test_envs.py
"""

import pytest

import numpy as np

import gym
import neurogym as ngym


def test_run(env, num_steps=100, verbose=False, **kwargs):
    """Test if one environment can at least be run."""
    if isinstance(env, str):
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')

    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

    tags = env.metadata.get('tags', [])
    all_tags = ngym.all_tags()
    for t in tags:
        if t not in all_tags:
            print('Warning: env has tag {:s} not in all_tags'.format(t))

    if verbose:
        print(env)

    return env


def test_print_all():
    """Test printing of all experiments."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ngym.all_envs()):
        total_count += 1
        print('')
        print('Test printing env: {:s}'.format(env_name))
        try:
            env = gym.make(env_name)
            print(env)
            print('Success')
            success_count += 1
        except BaseException as e:
            print('Failure')
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))


def test_run_all(verbose_success=False):
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ngym.all_envs(psychopy=True, collections=True)):
        total_count += 1

        # print('Running env: {:s}'.format(env_name))
        # env = test_run(env_name)
        try:
            test_run(env_name, verbose=verbose_success)
            # print('Success')
            # print(env)
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))


def test_trialenv_all():
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    hastrial_count = 0
    for env_name in sorted(ngym.all_envs()):
        if env_name in ['Combine-v0']:
            continue
        env = gym.make(env_name)
        if not isinstance(env, ngym.TrialEnv):
            continue
        total_count += 1

        print('Running env: {:s}'.format(env_name))
        try:
            env.new_trial()
            if env.trial is None:
                print('No trial is available after new_trial()')
            else:
                print('Success')
                hastrial_count += 1
            # print(env)
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))
    print('{:d}/{:d} envs have trial after new_trial'.format(hastrial_count,
                                                                  success_count))


def test_seeding_all():
    """Test if all environments can at least be run."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ngym.all_envs()):
        total_count += 1

        # print('Running env: {:s}'.format(env_name))
        # env = test_run(env_name)
        try:
            states1, rews1 = test_seeding(env_name, seed=0)
            states2, rews2 = test_seeding(env_name, seed=0)
            assert (states1 == states2).all(), 'states are not identical'
            assert (rews1 == rews2).all(), 'rewards are not identical'
            states1, rews1 = test_seeding(env_name, seed=0)
            states2, rews2 = test_seeding(env_name, seed=0)
            assert (states1 == states2).all(), 'states are not identical'
            assert (rews1 == rews2).all(), 'rewards are not identical'

            # print('Success')
            # print(env)
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))


def test_seeding(env, seed):
    """Test if environments are replicable."""
    if isinstance(env, str):
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')
    env.seed(seed=seed)
    env.reset()
    states_mat = []
    rew_mat = []
    env.action_space.seed(seed)
    for stp in range(100):
        state, rew, done, _ = env.step(env.action_space.sample())
        states_mat.append(state)
        rew_mat.append(rew)
        if done:
            env.reset()
    states_mat = np.array(states_mat)
    rew_mat = np.array(rew_mat)
    return states_mat, rew_mat


if __name__ == '__main__':
    test_run_all()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test environments.

All tests in this file can be run by running in command line
pytest test_envs.py
"""

import pytest

import numpy as np

import gym
import neurogym as ngym


try:
    import psychopy
    _have_psychopy = True
except ImportError as e:
    _have_psychopy = False

ENVS = ngym.all_envs(psychopy=_have_psychopy, contrib=True, collections=True)
# Envs without psychopy, TODO: check if contrib or collections include psychopy
ENVS_NOPSYCHOPY = ngym.all_envs(psychopy=False, contrib=True, collections=True)


def test_run(env=None, num_steps=100, verbose=False, **kwargs):
    """Test if one environment can at least be run."""
    if env is None:
        env = ngym.all_envs()[0]

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


def test_run_all(verbose_success=False):
    """Test if all environments can at least be run."""
    for env_name in sorted(ENVS):
        print(env_name)
        test_run(env_name, verbose=verbose_success)


def test_print_all():
    """Test printing of all experiments."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ENVS):
        total_count += 1
        print('')
        print('Test printing env: {:s}'.format(env_name))
        env = gym.make(env_name)
        print(env)


def test_trialenv(env=None, **kwargs):
    """Test if a TrialEnv is behaving correctly."""
    if env is None:
        env = ngym.all_envs()[0]

    if isinstance(env, str):
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')

    trial = env.new_trial()
    assert trial is not None, 'TrialEnv should return trial info dict ' + str(env)


def test_trialenv_all():
    """Test if all environments can at least be run."""
    for env_name in sorted(ENVS):
        env = gym.make(env_name)
        if not isinstance(env, ngym.TrialEnv):
            continue
        test_trialenv(env)


def test_seeding(env=None, seed=0):
    """Test if environments are replicable."""
    if env is None:
        env = ngym.all_envs()[0]

    if isinstance(env, str):
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')
    env.seed(seed)
    env.reset()
    ob_mat = []
    rew_mat = []
    act_mat = []
    for stp in range(100):
        action = env.action_space.sample()
        ob, rew, done, info = env.step(action)
        ob_mat.append(ob)
        rew_mat.append(rew)
        act_mat.append(action)
        if done:
            env.reset()
    ob_mat = np.array(ob_mat)
    rew_mat = np.array(rew_mat)
    act_mat = np.array(act_mat)
    return ob_mat, rew_mat, act_mat


def test_seeding_all():
    """Test if all environments can at least be run."""
    for env_name in sorted(ENVS_NOPSYCHOPY):
        # print('Running env: {:s}'.format(env_name))
        # env = test_run(env_name)
        obs1, rews1, acts1 = test_seeding(env_name, seed=0)
        obs2, rews2, acts2 = test_seeding(env_name, seed=0)
        assert (obs1 == obs2).all(), 'obs are not identical'
        assert (rews1 == rews2).all(), 'rewards are not identical'
        assert (acts1 == acts2).all(), 'rewards are not identical'
        # obs1, rews1 = test_seeding(env_name, seed=0)
        # obs2, rews2 = test_seeding(env_name, seed=0)
        # assert (obs1 == obs2).all(), 'obs are not identical'
        # assert (rews1 == rews2).all(), 'rewards are not identical'

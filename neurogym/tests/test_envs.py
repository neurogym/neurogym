#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test environments"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym as ngym


def test_run(env, verbose=False):
    """Main function for testing if an environment is healthy."""
    # TODO: consider removing this function and using directly _test_run()
    if isinstance(env, str):
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')
    _test_run(env)
    if verbose:
        print(env)
    return env


def _test_run(env):
    """Test if one environment can at least be run."""
    if isinstance(env, str):
        kwargs = {'dt': 20}
        env = gym.make(env, **kwargs)
    else:
        if not isinstance(env, gym.Env):
            raise ValueError('env must be a string or a gym.Env')

    env.reset()
    for stp in range(100):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

    tags = env.metadata.get('tags', [])
    all_tags = ngym.all_tags()
    for t in tags:
        if t not in all_tags:
            print('Warning: env has tag {:s} not in all_tags'.format(t))

    return env


def test_speed(env):
    """Test speed of an environment."""
    n_steps = 100000
    warmup_steps = 10000
    kwargs = {'dt': 20}

    if isinstance(env, str):
        env = gym.make(env, **kwargs)

    env.reset()
    for stp in range(warmup_steps):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()

    total_time = 0
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


def test_speed_all():
    """Test speed of all experiments."""
    for env_name in sorted(ngym.all_envs()):
        print('Running env: {:s}'.format(env_name))
        try:
            test_speed(env_name)
            print('Success')
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)


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
    for env_name in sorted(ngym.all_envs()):
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
                print('No self.trial is available after new_trial()')
            else:
                print('Success')
                hastrial_count += 1
            # print(env)
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)

    print('Success {:d}/{:d} envs'.format(success_count, total_count))
    print('{:d}/{:d} envs have self.trial after new_trial'.format(hastrial_count,
                                                                  success_count))


def test_plot(env_name, num_steps=500, kwargs={'dt': 100}):
    env = gym.make(env_name, **kwargs)

    env.reset()
    observations = []
    for stp in range(num_steps):
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
    plt.close('all')
    # test_seeding_all()
    # test_run_all()
    # test_speed_all()
    # test_trialenv_all()
    # test_print_all()
    # env_name = 'GoNogo-v0'
    env_name = 'PerceptualDecisionMaking-v0'
    # env_name = 'ContextDecisionMaking-v0'
    # env_name = 'NAltPerceptualDecisionMaking-v0'
    # env_name = 'DelayedMatchCategory-v0'
    # env_name = 'MemoryRecall-v0'
    # env_name = 'ReachingDelayResponse-v0'
    # test_run(env_name)
    kwargs = {'dt': 100, 'timing': {'fixation': ('constant', 200),
                                    'stimulus': ('constant', 200),
                                    'decision': ('constant', 200)}}
    test_plot(env_name, kwargs=kwargs, num_steps=20)
    # test_speed(env_name)
    # plot_env(env_name)

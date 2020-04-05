"""Test wrappers."""

import numpy as np
import gym
from gym import spaces
from gym.core import Wrapper
import matplotlib.pyplot as plt
import neurogym as ngym
from neurogym.wrappers import TrialHistory
from neurogym.wrappers import SideBias
from neurogym.wrappers import PassAction
from neurogym.wrappers import PassReward
from neurogym.wrappers import Identity
from neurogym.wrappers import Noise


def test_sidebias(env_name, num_steps=10000, verbose=False,
                  probs=[(.005, .005, .99),
                         (.005, .99, .005),
                         (.99, .005, .005)]):
    env = gym.make(env_name)
    env = SideBias(env, probs=probs, block_dur=10)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            if verbose:
                print('Ground truth', info['gt'])
                print('----------')
                print('Block', env.curr_block)
        # print(env.env.t)
        # print('')
        # print('Trial', env.unwrapped.num_tr)
        # print('Within trial time', env.unwrapped.t_ind)
        # print('Observation', obs)
        if done:
            env.reset()


def test_passaction(env_name, num_steps=10000, verbose=False, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = PassAction(env)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            print(obs)
            print(action)
            print('--------')

        if done:
            env.reset()


def test_passreward(env_name, num_steps=10000, verbose=False, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = PassReward(env)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            print(obs)
            print(rew)
            print('--------')
        if done:
            env.reset()


def test_noise(env_name, random_bhvr=0., wrapper=None, perf_th=None,
               num_steps=10000, verbose=False, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = Noise(env, perf_th=perf_th)
    if wrapper is not None:
        env = wrapper(env)
    env.reset()
    perf = []
    std_mat = []
    std_noise = 0
    for stp in range(num_steps):
        if np.random.rand() < random_bhvr + std_noise:
            action = env.action_space.sample()
        else:
            action = env.gt_now
        obs, rew, done, info = env.step(action)
        if 'std_noise' in info:
            std_noise = info['std_noise']
        if verbose:
            if info['new_trial']:
                perf.append(info['performance'])
                std_mat.append(std_noise)
                print(obs)
                print('--------')
        if done:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot([0, len(perf)], [perf_th, perf_th], '--')
        plt.plot(np.convolve(perf, np.ones((100,))/100))
        plt.subplot(2, 1, 2)
        plt.plot(std_mat)


def test_trialhist(env_name, num_steps=10000, probs=0.8, verbose=False):
    env = gym.make(env_name)
    env = TrialHistory(env, probs=probs, block_dur=50)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        if info['new_trial']:
            if verbose:
                print('Ground truth', info['gt'])
                print('------------')
                print('Block', env.curr_block)


def test_identity(env_name, num_steps=10000, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = Identity(env)
    env = Identity(env, id_='1')
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()


def test_all(test_fn):
    """Test speed of all experiments."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ngym.all_envs()):
        total_count += 1
        print('Running env: {:s} Wrapped with SideBias'.format(env_name))
        try:
            test_fn(env_name)
            print('Success')
            success_count += 1
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)
        print('')

    print('Success {:d}/{:d} envs'.format(success_count, total_count))


if __name__ == '__main__':
    plt.close('all')
    env_args = {'timing': {'fixation': ('constant', 100),
                           'stimulus': ('constant', 100),
                           'decision': ('constant', 100)}}
    # test_identity('Nothing-v0', num_steps=5)
    # test_passreward('PerceptualDecisionMaking-v0', num_steps=10, verbose=True,
    #                 **env_args)
    # test_passaction('PerceptualDecisionMaking-v0', num_steps=10, verbose=True,
    #                 **env_args)
    # test_noise('PerceptualDecisionMaking-v0', random_bhvr=0.,
    #            wrapper=PassAction, perf_th=0.7, num_steps=100000,
    #            verbose=True, **env_args)
    # test_trialhist('NAltPerceptualDecisionMaking-v0', num_steps=10000,
    #                verbose=True, probs=0.99)
    # test_sidebias('NAltPerceptualDecisionMaking-v0', num_steps=10000,
    #               verbose=True, probs=[(0, 0, 1), (0, 1, 0), (1, 0, 0)])

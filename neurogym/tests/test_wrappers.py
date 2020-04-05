"""Test wrappers."""

import numpy as np
import gym
# from gym import spaces
# from gym.core import Wrapper
import matplotlib.pyplot as plt
import neurogym as ngym
from neurogym.wrappers import TrialHistory
from neurogym.wrappers import SideBias
from neurogym.wrappers import PassAction
from neurogym.wrappers import PassReward
from neurogym.wrappers import Identity
from neurogym.wrappers import Noise
from neurogym.wrappers import CatchTrials
from neurogym.wrappers import ReactionTime


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
        if info['new_trial'] and verbose:
            print('Ground truth', info['gt'])
            print('----------')
            print('Block', env.curr_block)
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
        if info['new_trial'] and verbose:
            print('Ground truth', info['gt'])
            print('------------')
            print('Block', env.curr_block)


def test_catchtrials(env_name, num_steps=10000, verbose=False, catch_prob=0.1,
                     alt_rew=0):
    env = gym.make(env_name)
    env = CatchTrials(env, catch_prob=catch_prob, alt_rew=alt_rew)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial'] and verbose:
            print('Perfomance', (info['gt']-action) < 0.00001)
            print('catch-trial', info['catch_trial'])
            print('Reward', rew)
            print('-------------')
        if done:
            env.reset()


def test_reactiontime(env_name, num_steps=500, kwargs={'dt': 100},
                      ths=[-1, 1]):
    env = gym.make(env_name, **kwargs)
    env = ReactionTime(env)
    env.reset()
    observations = []
    obs_cum_mat = []
    actions = []
    new_trials = []
    obs_cum = 0
    for stp in range(num_steps):
        if obs_cum > ths[1]:
            action = 1
        elif obs_cum < ths[0]:
            action = 2
        else:
            action = 0
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            obs_cum = 0
        else:
            obs_cum += obs[1] - obs[2]
        observations.append(obs)
        actions.append(action)
        obs_cum_mat.append(obs_cum)
        new_trials.append(info['new_trial'])

    observations = np.array(observations)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(observations.T, aspect='auto')
    plt.subplot(3, 1, 2)
    plt.plot(actions)
    plt.subplot(3, 1, 3)
    plt.plot(obs_cum_mat)
    plt.plot([0, len(obs_cum_mat)], [ths[1], ths[1]], '--')
    plt.plot([0, len(obs_cum_mat)], [ths[0], ths[0]], '--')
    plt.show()


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
    # test_catchtrials('PerceptualDecisionMaking-v0', num_steps=10000,
    #                  verbose=True, catch_prob=0.5, alt_rew=0)
    test_reactiontime('PerceptualDecisionMaking-v0', num_steps=100)

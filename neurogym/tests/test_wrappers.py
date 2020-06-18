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
from neurogym.wrappers import TTLPulse
from neurogym.wrappers import TransferLearning
from neurogym.wrappers import Combine
from neurogym.wrappers import Variable_nch


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
    obs = env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            print(obs)
            print(rew)
            print('--------')
        if done:
            env.reset()


def test_ttlpulse(env_name, num_steps=10000, verbose=False, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = TTLPulse(env, periods=[['stimulus'], ['decision']])
    env.reset()
    obs_mat = []
    signals = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if verbose:
            obs_mat.append(obs)
            signals.append([info['signal_0'], info['signal_1']])
            print('--------')

        if done:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Observations')
        plt.imshow(np.array(obs_mat).T, aspect='auto')
        plt.subplot(2, 1, 2)
        plt.title('Pulses')
        plt.plot(signals)
        plt.xlim([-.5, num_steps-.5])


def test_transferLearning(num_steps=10000, verbose=False, **envArgs):
    task = 'GoNogo-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stimulus': ('constant', 100),
                                    'resp_delay': ('constant', 100),
                                    'decision': ('constant', 100)}}
    env1 = gym.make(task, **KWARGS)
    task = 'PerceptualDecisionMaking-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 100),
                                    'stimulus': ('constant', 100),
                                    'decision': ('constant', 100)}}
    env2 = gym.make(task, **KWARGS)
    env = TransferLearning([env1, env2], num_tr_per_task=[30], task_cue=True)

    env.reset()
    obs_mat = []
    signals = []
    rew_mat = []
    action_mat = []
    for stp in range(num_steps):
        # action = env.action_space.sample()
        action = 1
        obs, rew, done, info = env.step(action)
        if verbose:
            action_mat.append(action)
            rew_mat.append(rew)
            obs_mat.append(obs)
            signals.append([info['task']])

        if done:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title('Observations')
        plt.imshow(np.array(obs_mat).T, aspect='auto')
        plt.subplot(4, 1, 4)
        plt.title('Pulses')
        plt.plot(signals)
        plt.xlim([-.5, num_steps-.5])
        plt.subplot(4, 1, 3)
        plt.title('Actions')
        plt.plot(action_mat)
        plt.xlim([-.5, num_steps-.5])
        plt.subplot(4, 1, 2)
        plt.title('Reward')
        plt.plot(rew_mat)
        plt.xlim([-.5, num_steps-.5])


def test_combine(num_steps=10000, verbose=False, **envArgs):
    task = 'GoNogo-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stimulus': ('constant', 100),
                                    'resp_delay': ('constant', 100),
                                    'decision': ('constant', 100)}}
    env1 = gym.make(task, **KWARGS)
    task = 'DelayPairedAssociation-v0'
    KWARGS = {'dt': 100, 'timing': {'fixation': ('constant', 0),
                                    'stim1': ('constant', 100),
                                    'delay_btw_stim': ('constant', 500),
                                    'stim2': ('constant', 100),
                                    'delay_aft_stim': ('constant', 100),
                                    'decision': ('constant', 200)}}
    env2 = gym.make(task, **KWARGS)
    env = Combine(env=env1, distractor=env2, delay=100, dt=100, mix=(.3, .3, .4),
                  share_action_space=True, defaults=[0, 0], trial_cue=True)

    env.reset()
    obs_mat = []
    config_mat = []
    rew_mat = []
    action_mat = []
    for stp in range(num_steps):
        action = env.action_space.sample()
        # action = 1
        obs, rew, done, info = env.step(action)
        if verbose:
            action_mat.append(action)
            rew_mat.append(rew)
            obs_mat.append(obs)
            config_mat.append(info['task_type'])

        if done:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title('Observations')
        plt.imshow(np.array(obs_mat).T, aspect='auto')
        plt.subplot(4, 1, 4)
        plt.title('Trial configuration')
        plt.plot(config_mat)
        plt.xlim([-.5, num_steps-.5])
        plt.subplot(4, 1, 3)
        plt.title('Actions')
        plt.plot(action_mat)
        plt.xlim([-.5, num_steps-.5])
        plt.subplot(4, 1, 2)
        plt.title('Reward')
        plt.plot(rew_mat)
        plt.xlim([-.5, num_steps-.5])


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


def test_trialhist_and_variable_nch(env_name, num_steps=100000, probs=0.8,
                                    num_blocks=2, verbose=False, num_ch=4,
                                    variable_nch=True):
    env = gym.make(env_name, **{'n_ch': num_ch})
    env = TrialHistory(env, probs=probs, block_dur=200, num_blocks=num_blocks)
    if variable_nch:
        env = Variable_nch(env, block_nch=1000, blocks_probs=[0.1, 0.45, 0.45])
        transitions = np.zeros((num_ch-1, num_blocks, num_ch, num_ch))
    else:
        transitions = np.zeros((1, num_blocks, num_ch, num_ch))
    env.reset()
    blk = []
    gt = []
    nch = []
    prev_gt = 1
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        if info['new_trial'] and verbose:
            blk.append(info['curr_block'])
            gt.append(info['gt'])
            if variable_nch:
                nch.append(info['nch'])
                if len(nch) > 1 and nch[-1] == nch[-2] and blk[-1] == blk[-2]:
                    transitions[info['nch']-2, info['curr_block'], prev_gt,
                                info['gt']-1] += 1
            else:
                nch.append(num_ch)
                transitions[0, info['curr_block'], prev_gt, info['gt']-1] += 1
            prev_gt = info['gt']-1
    if verbose:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(blk[:20000], '-+')
        ax[0].plot(nch[:20000], '-+')
        ax[1].plot(gt[:20000], '-+')
        ch_mat = np.unique(nch)
        _, ax = plt.subplots(nrows=num_blocks, ncols=len(ch_mat))
        for ind_ch, ch in enumerate(ch_mat):
            for ind_blk in range(num_blocks):
                norm_counts = transitions[ind_ch, ind_blk, :, :]
                nxt_tr_counts = np.sum(norm_counts, axis=1).reshape((-1, 1))
                norm_counts = norm_counts / nxt_tr_counts
                ax[ind_blk][ind_ch].imshow(norm_counts)


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
    plt.xlim([-.5, len(actions)-0.5])
    plt.subplot(3, 1, 3)
    plt.plot(obs_cum_mat)
    plt.plot([0, len(obs_cum_mat)], [ths[1], ths[1]], '--')
    plt.plot([0, len(obs_cum_mat)], [ths[0], ths[0]], '--')
    plt.xlim([-.5, len(actions)-0.5])
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


def test_concat_wrpprs_th_vch_pssr_pssa(env_name, num_steps=100000, probs=0.8,
                                        num_blocks=16, verbose=False, num_ch=6,
                                        variable_nch=True):
    env = gym.make(env_name, **{'n_ch': num_ch})
    env = TrialHistory(env, probs=probs, num_blocks=num_blocks,
                       rand_blcks=True, blk_ch_prob=0.001)  # using random blocks!
    if variable_nch:
        env = Variable_nch(env, block_nch=1000, blocks_probs=[0.2, 0.2, 0.2,
                                                              0.2, 0.2])
        transitions = np.zeros((num_blocks, num_ch, num_ch))
    else:
        transitions = np.zeros((num_blocks, num_ch, num_ch))
    env = PassReward(env)
    env = PassAction(env)
    env.reset()
    num_tr_blks = np.zeros((num_blocks,))
    blk_id = []
    blk = []
    gt = []
    nch = []
    prev_gt = 1
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        if info['new_trial'] and verbose:
            blk_id, indx = check_blk_id(blk_id, info['curr_block'], num_blocks)
            # print(info['curr_block'])
            # print('-------------')
            blk.append(info['curr_block'])
            gt.append(info['gt'])
            if variable_nch:
                nch.append(info['nch'])
                if len(nch) > 2 and 2*[nch[-1]] == nch[-3:-1] and\
                   2*[blk[-1]] == blk[-3:-1] and\
                   indx != -1:
                    num_tr_blks[indx] += 1
                    transitions[indx, prev_gt, info['gt']-1] += 1
                    if prev_gt > info['nch'] or info['gt']-1 > info['nch']:
                        pass

            else:
                nch.append(num_ch)
                if blk[-1] == blk[-2] and indx != -1:
                    num_tr_blks[indx] += 1
                    transitions[indx, prev_gt, info['gt']-1] += 1
            prev_gt = info['gt']-1
    if verbose:
        print(blk_id)
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(np.array(blk[:20000])/(10**(num_ch-1)), '-+')
        ax[0].plot(nch[:20000], '-+')
        ax[1].plot(gt[:20000], '-+')
        num_cols_rows = int(np.sqrt(num_blocks))
        _, ax1 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax1 = ax1.flatten()
        _, ax2 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax2 = ax2.flatten()
        for ind_blk in range(num_blocks):
            norm_counts = transitions[ind_blk, :, :]
            ax1[ind_blk].imshow(norm_counts)
            ax1[ind_blk].set_title(str(blk_id[ind_blk]) +
                                   ' (N='+str(num_tr_blks[ind_blk])+')',
                                   fontsize=6)
            nxt_tr_counts = np.sum(norm_counts, axis=1).reshape((-1, 1))
            norm_counts = norm_counts / nxt_tr_counts
            ax2[ind_blk].imshow(norm_counts)
            ax2[ind_blk].set_title(str(blk_id[ind_blk]) +
                                   ' (N='+str(num_tr_blks[ind_blk])+')',
                                   fontsize=6)
    data = {'transitions': transitions, 'blk': blk, 'blk_id': blk_id, 'gt': gt,
            'nch': nch}
    return data


def check_blk_id(blk_id_mat, curr_blk, num_blk):
    if curr_blk in blk_id_mat:
        return blk_id_mat, np.argwhere(np.array(blk_id_mat) == curr_blk)
    elif len(blk_id_mat) < num_blk:
        blk_id_mat.append(curr_blk)
        return blk_id_mat, len(blk_id_mat)-1
    else:
        return blk_id_mat, -1


if __name__ == '__main__':
    plt.close('all')
    env_args = {'stim_scale': 10, 'timing': {'fixation': ('constant', 100),
                                             'stimulus': ('constant', 200),
                                             'decision': ('constant', 200)}}
    # test_identity('Nothing-v0', num_steps=5)
    # test_passreward('PerceptualDecisionMaking-v0', num_steps=10, verbose=True,
    #                 **env_args)
    # test_passaction('PerceptualDecisionMaking-v0', num_steps=10, verbose=True,
    #                 **env_args)
    # test_ttlpulse('PerceptualDecisionMaking-v0', num_steps=20, verbose=True,
    #               **env_args)
    # test_noise('PerceptualDecisionMaking-v0', random_bhvr=0.,
    #            wrapper=PassAction, perf_th=0.7, num_steps=100000,
    #            verbose=True, **env_args)
    # test_trialhist_and_variable_nch('NAltPerceptualDecisionMaking-v0',
    #                                 num_steps=1000000, verbose=True, probs=0.99,
    #                                 num_blocks=3)
    # test_sidebias('NAltPerceptualDecisionMaking-v0', num_steps=10000,
    #               verbose=True, probs=[(0, 0, 1), (0, 1, 0), (1, 0, 0)])
    # test_catchtrials('PerceptualDecisionMaking-v0', num_steps=10000,
    #                  verbose=True, catch_prob=0.5, alt_rew=0)
    # test_reactiontime('PerceptualDecisionMaking-v0', num_steps=100)
    # test_transferLearning(num_steps=200, verbose=True)
    # test_combine(num_steps=200, verbose=True)
    data = test_concat_wrpprs_th_vch_pssr_pssa('NAltPerceptualDecisionMaking-v0',
                                               num_steps=1000000, verbose=True,
                                               probs=0.99, num_blocks=16)

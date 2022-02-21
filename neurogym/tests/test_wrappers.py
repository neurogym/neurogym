"""Test wrappers."""

import numpy as np
import gym
import sys
# from gym import spaces
# from gym.core import Wrapper
import matplotlib.pyplot as plt
import neurogym as ngym
from neurogym.wrappers import SideBias
from neurogym.wrappers import PassAction
from neurogym.wrappers import PassReward
from neurogym.wrappers import Noise
from neurogym.wrappers import ReactionTime


def test_sidebias(env_name='NAltPerceptualDecisionMaking-v0', num_steps=100000,
                  verbose=False, num_ch=3, margin=0.01, blk_dur=10,
                  probs=[(.005, .005, .99), (.005, .99, .005), (.99, .005, .005)]):
    """
    Test side_bias wrapper.

    The side-bias wrapper allows specifying the probabilities for each of the
    existing choices to be correct. These probabilities can varied in blocks of
    duration blk_dur.

    Parameters
    ----------
    env_name : ngym.env, optional
        enviroment to wrap. The default is 'NAltPerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000000)
    verbose : boolean, optional
        whether to print ground truth and block (False)
    num_ch : int, optional
        number of choices (3)
    blk_dur : int, optional
        duration (in trials) of the blocks (10)
    margin : float, optional
        margin allow when comparing provided and obtained ground truth
        probabilities (0.01)
    probs : list, optional
        ground truth probabilities for each block.
        For example: [(.005, .005, .99), (.005, .99, .005), (.99, .005, .005)],
        corresponds to 3 blocks with each of them giving 0.99 probabilitiy to
        ground truth 3, 2 and 1, respectively.

    Returns
    -------
    None.

    """
    env_args['n_ch'] = num_ch
    env = gym.make(env_name, **env_args)
    env = SideBias(env, probs=probs, block_dur=blk_dur)
    env.reset()
    probs_mat = np.zeros((len(probs), num_ch))
    block = env.curr_block
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            probs_mat[block, info['gt']-1] += 1
            block = env.curr_block
            if verbose:
                print('Ground truth', info['gt'])
                print('----------')
                print('Block', block)
        if done:
            env.reset()
    probs_mat = probs_mat/np.sum(probs_mat, axis=1)
    assert np.mean(np.abs(probs-probs_mat)) < margin, 'Probs provided ' +\
        str(probs)+' probs. obtained '+str(probs_mat)
    print('-----')
    print('Side bias wrapper OK')


def test_passaction(env_name='PerceptualDecisionMaking-v0', num_steps=1000,
                    verbose=True):
    """
    Test pass-action wrapper.

    TODO: explain wrapper
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and action (False)

    Returns
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassAction(env)
    env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        assert obs[-1] == action, 'Previous action is not part of observation'
        if verbose:
            print(obs)
            print(action)
            print('--------')

        if done:
            env.reset()


def test_passreward(env_name='PerceptualDecisionMaking-v0', num_steps=1000,
                    verbose=False):
    """
    Test pass-reward wrapper.
    TODO: explain wrapper
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassReward(env)
    obs = env.reset()
    for stp in range(num_steps):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        assert obs[-1] == rew, 'Previous reward is not part of observation'
        if verbose:
            print(obs)
            print(rew)
            print('--------')
        if done:
            env.reset()


def test_reactiontime(env_name='PerceptualDecisionMaking-v0', num_steps=10000,
                      urgency=-0.1, ths=[-.5, .5], verbose=True):
    """
    Test reaction-time wrapper.

    The reaction-time wrapper allows converting a fix duration task into a reaction
    time task. It also allows addding a fix (negative) quantity (urgency) to force
    the network to respond quickly.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    urgency : float, optional
        float value added to the reward (-0.1)
    verbose : boolean, optional
        whether to print observation and reward (False)
    ths : list, optional
        list containing the threholds to make a decision ([-.5, .5])

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 2000, 'decision': 200}}
    env = gym.make(env_name, **env_args)
    env = ReactionTime(env, urgency=urgency)
    env.reset()
    if verbose:
        observations = []
        obs_cum_mat = []
        actions = []
        new_trials = []
        reward = []
    obs_cum = 0
    end_of_trial = False
    step = 0
    for stp in range(num_steps):
        if obs_cum > ths[1]:
            action = 1
        elif obs_cum < ths[0]:
            action = 2
        else:
            action = 0
        end_of_trial = True if action != 0 else False
        obs, rew, done, info = env.step(action)
        if info['new_trial']:
            step = 0
            obs_cum = 0
            end_of_trial = False
        else:
            step += 1
            assert not end_of_trial, 'Trial still on after making a decision'
            obs_cum += obs[1] - obs[2]
        if verbose:
            observations.append(obs)
            actions.append(action)
            obs_cum_mat.append(obs_cum)
            new_trials.append(info['new_trial'])
            reward.append(rew)
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect='auto')
        ax[1].plot(actions, label='Actions')
        ax[1].plot(new_trials, '--', label='New trial')
        ax[1].set_xlim([-.5, len(actions)-0.5])
        ax[1].legend()
        ax[2].plot(obs_cum_mat, label='cum. observation')
        ax[2].plot([0, len(obs_cum_mat)], [ths[1], ths[1]], '--', label='upper th')
        ax[2].plot([0, len(obs_cum_mat)], [ths[0], ths[0]], '--', label='lower th')
        ax[2].set_xlim([-.5, len(actions)-0.5])
        ax[3].plot(reward, label='reward')
        ax[3].set_xlim([-.5, len(actions)-0.5])


def test_noise(env='PerceptualDecisionMaking-v0', margin=0.01, perf_th=0.7,
               num_steps=100000, verbose=True):
    """
    Test noise wrapper.

    The noise wrapper allows adding noise to the full observation received by the
    network. It also offers the option of fixxing a specific target performance
    that the wrapper will assure by modulating the magnitude of the noise added.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)
    margin : float, optional
        margin allowed when comparing actual and expected performances (0.01)
    perf_th : float, optional
        target performance for the noise wrapper (0.7)

    Returns
    -------
    None.

    """
    env_args = {'timing': {'fixation': 100, 'stimulus': 200, 'decision': 200}}
    env = gym.make(env, **env_args)
    env = Noise(env, std_noise=1)
    obs = env.reset()
    perf = []
    std_mat = []
    std_noise = 0
    for stp in range(num_steps):
        if np.random.rand() < std_noise:
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


def check_blk_id(blk_id_mat, curr_blk, num_blk, sel_chs):
    # translate transitions t.i.a. selected choices
    # curr_blk_indx = list(curr_blk.replace('-', ''))
    # curr_blk_indx = [sel_chs[int(x)-1] for x in curr_blk_indx]
    # curr_blk = '-'.join([str(x) for x in curr_blk_indx])
    if curr_blk in blk_id_mat:
        return blk_id_mat, np.argwhere(np.array(blk_id_mat) == curr_blk)
    elif len(blk_id_mat) < num_blk:
        blk_id_mat.append(curr_blk)
        return blk_id_mat, len(blk_id_mat)-1
    else:
        return blk_id_mat, -1


if __name__ == '__main__':
    plt.close('all')
    env_args = {'stim_scale': 10, 'timing': {'fixation': 100,
                                             'stimulus': 200,
                                             'decision': 200}}
    # test_identity('Null-v0', num_steps=5)
    # test_reactiontime()
    test_passreward()
    test_noise()
    test_sidebias()
    test_passaction()

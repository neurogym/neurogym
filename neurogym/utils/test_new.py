# TODO: Remove this file

raise ValueError('Deprecated file')

import matplotlib
import numpy as np
import sys
from os.path import expanduser
home = expanduser("~")
sys.path.append(home)
sys.path.append(home + '/neurogym')
sys.path.append(home + '/gym')
import gym
from neurogym.wrappers import trial_hist
from neurogym.wrappers import reaction_time
from neurogym.wrappers import combine
from neurogym.wrappers import pass_reward
from neurogym.wrappers import pass_action
from neurogym.wrappers import side_bias
from neurogym.wrappers import manage_data as md
display_mode = True
if display_mode:
    matplotlib.use('Qt5Agg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    aadhf = argparse.ArgumentDefaultsHelpFormatter
    return argparse.ArgumentParser(formatter_class=aadhf)


def neuro_arg_parser():
    """
    Create an argparse.ArgumentParser for neuro environments
    """
    parser = arg_parser()
    parser.add_argument('--pass_reward', '-pr',
                        help='whether to pass the prev. reward with obs',
                        type=bool, default=False)
    parser.add_argument('--pass_action', '-pa',
                        help='whether to pass the prev. action with obs',
                        type=bool, default=False)
    parser.add_argument('--combine',
                        help='whether to combine with another task',
                        type=bool, default=False)
    parser.add_argument('--trial_hist',
                        help='whether to introduce trial history',
                        type=bool, default=False)
    parser.add_argument('--side_bias',
                        help='whether to introduce side bias',
                        type=bool, default=False)
    parser.add_argument('--reaction_time', '-rt',
                        help='whether to make the task a reaction-time task',
                        type=bool, default=False)
    parser.add_argument('--env', help='primary task',
                        type=str, default='PostDecisionWager-v0')
    parser.add_argument('--env2', help='secondary task (if combine is True)',
                        type=str, default='GoNogo-v0')
    parser.add_argument('--n_steps',
                        help='number of steps',
                        type=int, default=400)
    parser.add_argument('--delay',
                        help='delay with which env2 will start',
                        type=int, default=500)
    parser.add_argument('--bl_dur',
                        help='dur. of block in the trial-hist wrappr (trials)',
                        type=int, nargs='+', default=(200,))
    parser.add_argument('--stim_scale', help='allows scaling stimulus evidence',
                        type=float, nargs='+', default=(1.,))
    parser.add_argument('--plot', help='show figure',
                        type=bool, default=True)
    parser.add_argument('--dt', help='timestep', type=int, default=100)
    return parser


def main(args):
    arg_pars = neuro_arg_parser()
    args, unknown_args = arg_pars.parse_known_args(args)
    env = args.env
    pass_rew = args.pass_reward
    pass_act = args.pass_action
    tr_hist = args.trial_hist
    react_t = args.reaction_time
    s_bias = args.side_bias
    comb = args.combine
    if combine:
        env2 = args.env2
        delay = args.delay
    plot_fig = args.plot
    num_steps = args.n_steps  # [1e9]
    dt = args.dt

    # task
    print('Making ' + env + ' task')
    env = gym.make(env, **{'dt': dt})
    # wrappers
    print('xxxxxx')
    print('Wrappers')
    if tr_hist:
        print('trial history')
        env = trial_hist.TrialHistory(env)
    if react_t:
        print('reaction time')
        env = reaction_time.ReactionTime(env)
    if comb:
        print('combine with ' + env2)
        env_extra = gym.make(env2, **{'dt': dt})
        # delay is in ms
        env = combine.combine(dt=dt, env1=env, env2=env_extra, delay=delay)
    if pass_rew:
        print('pass reward')
        env = pass_reward.PassReward(env)
    if pass_act:
        print('pass action')
        env = pass_action.PassAction(env)
    if s_bias:
        print('side bias')
        env = side_bias.SideBias(env)

    # save/render data wrapper
    env = md.manage_data(env, plt_tr=False)
    env.seed(0)
    env.action_space.seed(0)
    observations = []
    rewards = []
    actions = []
    actions_end_of_trial = []
    gt = []
    config_mat = []
    for stp in range(int(num_steps)):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        if done:
            env.reset()
        observations.append(obs)
        if info['new_trial']:
            actions_end_of_trial.append(action)
        else:
            actions_end_of_trial.append(-1)
        rewards.append(rew)
        actions.append(action)
        gt.append(info['gt'])
        if 'config' in info.keys():
            config_mat.append(info['config'])
        else:
            config_mat.append([0, 0])

    if plot_fig:
        rows = 4
        obs = np.array(observations)
        plt.figure()
        plt.subplot(rows, 1, 1)
        plt.imshow(obs.T, aspect='auto')
        plt.title('observations')
        plt.subplot(rows, 1, 2)
        config_mat = np.array(config_mat)
        plt.imshow(config_mat.T, aspect='auto')
        plt.subplot(rows, 1, 3)
        plt.plot(actions, marker='+')
        plt.plot(actions_end_of_trial, '--')
        gt = np.array(gt)
        plt.plot(np.argmax(gt, axis=1), 'r')
        print(np.sum(np.argmax(gt, axis=1) == 2))
        print(np.sum(np.argmax(gt, axis=1) == 1))
        # aux = np.argmax(obs, axis=1)
        # aux[np.sum(obs, axis=1) == 0] = -1
        # plt.plot(aux, '--k')
        plt.title('actions')
        plt.xlim([-0.5, len(rewards)+0.5])
        plt.subplot(rows, 1, 4)
        plt.plot(rewards, 'r')
        plt.title('reward')
        plt.xlim([-0.5, len(rewards)+0.5])
        plt.show()


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")
sys.path.append(home + '/neurogym')
sys.path.append(home + '/mm5514/')
from neurogym.ops import utils as ut
from neurogym.ops import put_together_files as ptf
from neurogym.ops import results_summary as res_summ
import call_function as cf
display_mode = True
DPI = 400
num_trials_back = 6
n_exps_fig_2_ccn = 50  # 217
acr_tr_per = 100000
acr_train_step = 40000
letters_size = 12
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
gris = np.array((.5, .5, 0.5))
colores = np.concatenate((azul.reshape((1, 3)), rojo.reshape((1, 3)),
                          verde.reshape((1, 3)), morado.reshape((1, 3)),
                          naranja.reshape((1, 3))), axis=0)


def plot_learning(reward, w_conv=200, legend=True, lw=0.1, title=''):
    """
    plots RNN and ideal observer performances.
    The function assumes that a figure has been created
    before it is called.
    """
    num_trials = reward.shape[0]

    # save the mean rewards
    RNN_perf = np.mean(reward[2000:].flatten())

    # plot smoothed reward
    reward_smoothed = np.convolve(reward, np.ones((w_conv,))/w_conv,
                                  mode='valid')
    reward_smoothed = reward_smoothed[0::w_conv]
    plt.plot(np.linspace(0, num_trials, reward_smoothed.shape[0]),
             reward_smoothed, lw=lw,
             label='RNN perf. (' + str(round(RNN_perf, 3)) + ')', alpha=0.5)
             # color=(0.39, 0.39, 0.39)
    # plot 0.25, 0.5 and 0.75 reward lines
    plot_fractions([0, reward.shape[0]])
    if title != '':
        plt.title(title)
    plt.xlabel('trials')
    if legend:
        plt.legend()


def plot_fractions(lims):
    """
    plot dashed lines for 0.25, 0.5 and 0.75
    """
    plt.plot(lims, [0.25, 0.25], '--k', lw=0.25)
    plt.plot(lims, [0.5, 0.5], '--k', lw=0.25)
    plt.plot(lims, [0.75, 0.75], '--k', lw=0.25)
    plt.xlim(lims[0], lims[1])


def load_behavioral_data(file):
    """
    loads behavioral data and get relevant info from it
    """
    data = np.load(file)
    choice = data['choice']
    stimulus = data['stimulus']
    correct_side = data['correct_side']
    reward = data['reward']
    performance = (choice == correct_side)
    evidence = stimulus[:, 1] - stimulus[:, 2]
    return choice, correct_side, performance, evidence, reward


def get_simulation_vars(file='/home/linux/network_data_492999.npz', fig=False,
                        n_envs=12, env=0, num_steps=100, obs_size=4,
                        num_units=128, num_act=3, num_steps_fig=200, start=0,
                        save_folder='', title=''):
    """
    given a file produced by the A2C algorithm in baselines, it returns the
    states, rewards, actions, stimulus evidence and new trials vectors
    corresponding to a given environment
    """
    data = np.load(file)
    # states
    states = data['states'][:, :, env, :]

    states = np.reshape(np.transpose(states, (2, 0, 1)),
                        (states.shape[2], np.prod(states.shape[0:2])))
    # actions
    # separate into diff. envs
    print('observation shape: ' + str(data['obs'].shape))
    print(n_envs)
    actions = np.reshape(data['actions'], (-1, n_envs, num_steps))
    # select env.
    actions = actions[:, env, :]
    # flatten
    actions = actions.flatten()
    # actions = np.concatenate((np.array([0]), actions[:-1]))
    # obs and rewards (rewards are passed as part of the observation)
    obs = np.reshape(data['obs'], (-1, n_envs, num_steps, obs_size))
    obs = obs[:, env, :, :]
    obs = np.reshape(np.transpose(obs, (2, 0, 1)),
                     (obs.shape[2], np.prod(obs.shape[0:2])))
    rewards = obs[obs_size-1, :]  # TODO: make this index a parameter
    rewards = rewards.flatten()
    ev = obs[1, :] - obs[2, :]
    ev = ev.flatten()
    # trials
    trials = np.reshape(data['trials'], (-1, n_envs, num_steps))
    trials = trials[:, env, :]
    trials = trials.flatten()
    trials = np.concatenate((np.array([0]), trials[:-1]))
    if 'gt' in data.keys():
        # get ground truth
        gt = np.reshape(data['gt'], (-1, n_envs, num_steps, num_act))
        gt = gt[:, env, :, :]
        gt = np.reshape(np.transpose(gt, (2, 0, 1)),
                        (gt.shape[2], np.prod(gt.shape[0:2])))
    else:
        gt = []

    if 'pi' in data.keys():
        # separate into diff. envs
        pi = np.reshape(data['pi'], (-1, n_envs, num_steps, num_act))
        # select env.
        pi = pi[:, env, :]
        # flatten
        pi = pi.flatten()
    else:
        pi = []
    if fig:
        rows = 4
        cols = 1
        lw = 1
        gris = (.6, .6, .6)
        trials_temp = trials[start:start+num_steps_fig]
        tr_time = np.where(trials_temp == 1)[0] + 0.5
        f = ut.get_fig(display_mode, font=12)
        # FIGURE
#        # states
        plt.subplot(rows, cols, 2)
        maxs = np.max(states, axis=1).reshape((num_units, 1))
        states_norm = states / maxs
        states_norm[np.where(maxs == 0), :] = 0
        plt.imshow(states_norm[:, start:start+num_steps_fig], aspect='auto')
        for ind_tr in range(len(tr_time)):
            plt.plot(np.ones((2,))*tr_time[ind_tr], [-0.5, num_units-0.5],
                     '--', color=gris, lw=lw)
        plt.ylabel('Neurons activity')
        plt.yticks([])
        plt.xticks([])
        # actions and gt
        if len(gt) > 0:
            gt_temp = np.argmax(gt, axis=0)
        else:
            gt_temp = []
        plt.subplot(rows, cols, 3)
        for ind_tr in range(len(tr_time)):
            plt.plot(np.ones((2,))*tr_time[ind_tr], [0, 2], '--',
                     color=gris, lw=lw)
        plt.plot(actions[start:start+num_steps_fig], '-+', lw=lw,
                 color='k')  # colores[2, :])
        plt.plot(gt_temp[start:start+num_steps_fig], '--+', lw=lw,
                 color=colores[4, :])
        plt.xlim([-0.5, num_steps_fig-0.5])
        plt.ylabel('Action')
        plt.xticks([])
        plt.yticks([0, 1, 2])
        # obs
        plt.subplot(rows, cols, 1)
        plt.imshow(obs[:, start:start+num_steps_fig], aspect='auto')
        for ind_tr in range(len(tr_time)):
            plt.plot(np.ones((2,))*tr_time[ind_tr], [-0.5, obs_size-0.5], '--',
                     color=gris, lw=lw)
        plt.ylabel('Observation')
        plt.yticks([])
        plt.xticks([])
        if title != '':
            plt.title(title)
        # rewards
        plt.subplot(rows, cols, 4)
        for ind_tr in range(len(tr_time)):
            plt.plot(np.ones((2,))*tr_time[ind_tr], [0, 1], '--',
                     color=gris, lw=lw)
        plt.plot(rewards[start:start+num_steps_fig], '-+', lw=lw,
                 color='k')
        plt.xlim([-0.5, num_steps_fig-0.5])
        plt.ylabel('Reward')
        plt.xlabel('Timesteps (a.u)')
        plt.yticks([0, 1])
        if save_folder != '':
            f.savefig(save_folder+'/experiment_structure.svg', dpi=DPI,
                      bbox_inches='tight')

    return states, rewards, actions, ev, trials, gt, pi


if __name__ == '__main__':
    plt.close('all')
    num_steps = 100
    test = 'tests2'
    get_simulation_vars(file='/home/molano/priors/' + test + '/' +
                        'network_data_99.npz', fig=True,
                        n_envs=1, env=0, num_steps=20, obs_size=3,
                        num_units=64, num_act=3, num_steps_fig=num_steps,
                        start=0, save_folder='', title=test)
    asdasd

#    get_simulation_vars(file='/home/molano/neurogym/results/Romo/' +
#                        'a2c_Romo_t_100_200_200_200_200_200_PR_PA_' +
#                        'cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20' +
#                        '_ne_40_nu_64_ev_0.5_a_0.1_n_0_434044' +
#                        '/network_data_124999.npz', fig=True,
#                        n_envs=40, env=0, num_steps=20, obs_size=4,
#                        num_units=64, num_act=3, num_steps_fig=num_steps,
#                        start=0,
#                        save_folder='', title='pdwager early')
#    asdasd
#    get_simulation_vars(file='/home/molano/neurogym/results/example/' +
#                        'network_data_899.npz', fig=True,
#                        n_envs=1, env=0, num_steps=20, obs_size=4,
#                        num_units=64, num_act=3, num_steps_fig=num_steps,
#                        start=0, title='',
#                        save_folder='/home/molano/neurogym/results/example/')
#    asdasd
#    get_simulation_vars(file='/home/molano/neurogym/results/pdWager/' +
#                        'a2c_pdWager_t_200_300_400_500_200_300_400_200' +
#                        '_200_200_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_' +
#                        'lrs_c_g_0.8_b_20_ne_40_nu_64_ev_0.5_a_0.1_n_0_' +
#                        '856875/network_data_24999.npz', fig=True,
#                        n_envs=40, env=0, num_steps=20, obs_size=6,
#                        num_units=64, num_act=3, num_steps_fig=num_steps,
#                        start=0,
#                        save_folder='', title='pdwager early')
#    get_simulation_vars(file='/home/molano/neurogym/results/pdWager/' +
#                        'a2c_pdWager_t_200_300_400_500_200_300_400_200' +
#                        '_200_200_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_' +
#                        'lrs_c_g_0.8_b_20_ne_40_nu_64_ev_0.5_a_0.1_n_0_' +
#                        '856875/network_data_124999.npz', fig=True,
#                        n_envs=40, env=0, num_steps=20, obs_size=6,
#                        num_units=64, num_act=3, num_steps_fig=num_steps,
#                        start=490000,
#                        save_folder='', title='pdwager late')
#    asdsd
#    get_simulation_vars(file='/home/molano/neurogym/results/padoaSch/' +
#                        'a2c_padoaSch_t_200_200_400_200_PR_PA_cont_rnn_' +
#                        'ec_0.05_lr_0.001_lrs_c_g_0.8_b_20_ne_40_nu_64_' +
#                        'ev_0.5_a_0.1_n_0_853669/network_data_124999.npz',
#                        fig=True,
#                        n_envs=40, env=0, num_steps=20, obs_size=9,
#                        num_units=64, num_act=3, num_steps_fig=num_steps,
#                        start=490000,
#                        save_folder='', title='padoaSch')
    per = 50000
    exps_list = ['padoaSch', 'pdWager', 'priors', 'DelayedMatchSample',
                 'DawTwoStep', 'RDM', 'GNG', 'dual_task', 'Mante', 'DPA']
    sbplt_pos = [1, 6, 10, 2, 3, 5, 7, 9, 4, 8]
    exps_titles = ['Value-Based', 'Post-Dec. Wager', 'RDM + Trial Hist.',
                   'Delayed-Match-Sample', 'Two-Step',
                   'RDM', 'Go/No-Go', 'Dual-Task', 'Context', 'DPA']
    xaxis_lim = np.array([45, 15, 30, 28, 25, 20, 25, 15, 20, 25])*10000
    main_folder = '/home/molano/neurogym/results/'
    exps = glob.glob(main_folder + '*')
    print(exps)
    f = ut.get_fig(font=8, figsize=(8, 3))
    counter = 0
    for ind_exp in range(len(exps)):
        folder_name = exps[ind_exp] + '/'
        experiment = os.path.basename(os.path.normpath(folder_name))
        if experiment in exps_list:
            print('-----------------------')
            print(experiment)
            files = glob.glob(folder_name + '*')
            plt.subplot(2, 5, sbplt_pos[counter])
            plt.title(experiment)
            leg_flag = False
            title = exps_titles[counter]
            at_least_one = False
            for ind_f in range(len(files)):
                file = files[ind_f] + '/bhvr_data_all.npz'
                data_flag = ptf.put_together_files(files[ind_f],
                                                   min_num_trials=per)
                if data_flag:
                    _, _, _, _, reward =\
                        load_behavioral_data(file)
                    plot_learning(reward, w_conv=200, legend=leg_flag,
                                  title=title)
                    leg_flag = False
                    title = ''
                    at_least_one = True
            if at_least_one:
                print('some data')
            ax = plt.gca()
            if sbplt_pos[counter] == 6:
                ax.set_yticks([0, .25, .5, .75, 1.])
                ax.set_ylabel('reward per trial')
            elif sbplt_pos[counter] == 1:
                ax.set_yticks([0, .25, .5, .75, 1.])
                ax.set_ylabel('reward per trial')
                ax.set_xlabel('')
            else:
                ax.set_yticks([])
                ax.set_xlabel('')
            ax.set_ylim([0, 1])
            ax.set_xlim([0, xaxis_lim[sbplt_pos[counter]-1]])
            ax.set_xticks([0, xaxis_lim[sbplt_pos[counter]-1]])
            counter += 1
    f.savefig('/home/molano/CNS_neurogym_poster/performances.svg', dpi=DPI,
              bbox_inches='tight')

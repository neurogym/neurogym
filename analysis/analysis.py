#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from scipy.special import erf
import os
import glob
import numpy as np
import scipy.stats as sstats
from scipy.optimize import curve_fit
import matplotlib
import json
from os.path import expanduser
home = expanduser("~")
sys.path.append(home + '/neurogym')
sys.path.append(home + '/mm5514/')
from neurogym.ops import utils as ut
from neurogym.ops import put_together_files as ptf
from neurogym.ops import results_summary as res_summ
import call_function as cf
matplotlib.use('Agg')  # Qt5Agg
import matplotlib.pyplot as plt
display_mode = True
DPI = 400
num_trials_back = 6

############################################
# AUXLIARY FUNCTIONS
############################################


def get_repetitions(mat):
    """
    makes diff of the input vector, mat, to obtain the repetition vector X,
    i.e. X will be 1 at t if the value of mat at t is equal to that at t-1
    """
    mat = mat.flatten()
    values = np.unique(mat)
    rand_ch = np.array(np.random.choice(values, size=(1,)))
    repeat_choice = np.concatenate((rand_ch, mat))
    return (np.diff(repeat_choice) == 0)*1


def get_transition_mat(choice, times=None, num_steps=None, conv_window=5):
    """
    convolves the repetition vector obtained from choice to get a count of the
    number of repetitions in the last N trials (N = conv_window),
    **without taking into accountthe current trial**.
    It can return the whole vector (times!=None) or just the outcomes
    (i.e. transition.shape==choice.shape)
    """
    # selectivity to transition probability
    repeat = get_repetitions(choice)
    transition = np.convolve(repeat, np.ones((conv_window,)),
                             mode='full')[0:-conv_window+1]
    transition_ev = np.concatenate((np.array([0]), transition[:-1]))
    transition_ev -= conv_window/2
    if times is not None:
        trans_mat = np.zeros((num_steps,))
        for ind_t in range(times.shape[0]):
            trans_mat[times[ind_t]] = transition_ev[ind_t]
        return trans_mat
    else:
        return transition_ev


def bias_calculation(choice, ev, mask):
    """
    compute repating bias given the choice of the network, the stimulus
    evidence and a mask indicating the trials on which the bias should
    be computed
    """
    # associate invalid trials (network fixates) with incorrect choice
    choice[choice == 0] = ev[choice == 0] > 0
    repeat = get_repetitions(choice)
    # choice_repeating is just the original right_choice mat
    # but shifted one element to the left.
    choice_repeating = np.concatenate(
        (np.array(np.random.choice([0, 1])).reshape(1, ),
         choice[:-1]))
    # the rep. evidence is the original evidence with a negative sign
    # if the repeating side is the left one
    rep_ev = ev *\
        (-1)**(choice_repeating == 2)
    rep_ev_mask = rep_ev[mask]
    repeat_mask = repeat[mask]
    try:
        popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask, repeat_mask,
                               maxfev=10000)
    except:
        popt = [0, 0]
        pcov = 0
        print('no fitting')
    return popt, pcov


def build_block_mat(shape, block_dur, corr_side=None):
    """
    create rep/alt blocks based on the block duration. If the correct side is
    provided, the blocks show the actual repetition probability
    """
    # build rep. prob vector
    rp_mat = np.zeros(shape)
    a = np.arange(shape[0])
    b = np.floor(a/block_dur)
    rp_mat[b % 2 == 0] = 1
    if corr_side is not None:
        rep_mat = get_repetitions(corr_side)
        rp_mat[rp_mat == 0] = np.round(np.mean(rep_mat[rp_mat == 0]), 1)
        rp_mat[rp_mat == 1] = np.round(np.mean(rep_mat[rp_mat == 1]), 1)
    return rp_mat


def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    builds probit function. If piR/piL are not zero, it will provide the lapse
    rate probit fit.
    """
    piR = 0
    piL = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def probit(x, beta, alpha):
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


############################################
# PLOT AUXLIARY FUNCTIONS
############################################


def plot_time_event(times=[0]):
    """
    plots vertical line at the times specified by times
    """
    ax = plt.gca()
    ylim = ax.get_ylim()
    for ind_t in times:
        plt.plot([ind_t, ind_t], ylim, '--', color=(.7, .7, .7))


def plot_fractions(lims):
    """
    plot dashed lines for 0.25, 0.5 and 0.75
    """
    plt.plot(lims, [0.25, 0.25], '--k', lw=0.25)
    plt.plot(lims, [0.5, 0.5], '--k', lw=0.25)
    plt.plot(lims, [0.75, 0.75], '--k', lw=0.25)
    plt.xlim(lims[0], lims[1])


def plot_dashed_lines(minimo, maximo, value=0.5):
    plt.plot([0, 0], [0, 1], '--k', lw=0.2)
    plt.plot([minimo, maximo], [value, value], '--k', lw=0.2)


def plot_lines(x_max, y_value):
    plt.plot([0, x_max], [y_value, y_value], '--', color=(.7, .7, .7))


############################################
# NEURAL ANALYSIS
############################################


def get_simulation_vars(file='/home/linux/network_data_492999.npz', fig=False,
                        n_envs=12, env=0, num_steps=100, obs_size=4,
                        num_units=128, num_act=3):
    """
    given a file produced by the A2C algorithm in baselines, it returns the
    states, rewards, actions, stimulus evidence and new trials vectors
    corresponding to a given environment
    """
    data = np.load(file)
    rows = 5
    cols = 1
    # states
    states = data['states'][:, :, env, :]

    states = np.reshape(np.transpose(states, (2, 0, 1)),
                        (states.shape[2], np.prod(states.shape[0:2])))
    # actions
    # separate into diff. envs
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
    rewards = obs[3, :]
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
        gt = np.reshape(data['gt'], (-1, n_envs, num_steps, 3))
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

        num_steps = 200
        ut.get_fig(display_mode)
        # FIGURE
        # states
        plt.subplot(rows, cols, 2)
        maxs = np.max(states, axis=1).reshape((num_units, 1))
        states_norm = states / maxs
        states_norm[np.where(maxs == 0), :] = 0
        plt.imshow(states_norm[:, 0:num_steps], aspect='auto')
        # actions
        plt.subplot(rows, cols, 3)
        plt.plot(actions[0:num_steps], '-+')
        plt.xlim([-0.5, num_steps-0.5])
        # obs
        plt.subplot(rows, cols, 1)
        plt.imshow(obs[:, 0:num_steps], aspect='auto')
        # trials
        plt.subplot(rows, cols, 4)
        plt.plot(trials[0:num_steps], '-+')
        plt.xlim([-0.5, num_steps-0.5])
        # gt
        plt.subplot(rows, cols, 5)
        plt.imshow(gt[:, 0:num_steps], aspect='auto')

    return states, rewards, actions, ev, trials, gt, pi


def neuron_selectivity(activity, feature, all_times, feat_bin=None,
                       window=(-5, 10), av_across_time=False,
                       prev_tr=False):
    """
    computes the average activity conditioned on feature at windows around the
    times given by all_times. f feat_bin is not none, it bins the feature
    values. If av_across_time is True it averages the activity across time.
    It also returns a significance mat computed using ranksums for all times
    and feature values.
    """
    times = all_times[np.logical_and(all_times > np.abs(window[0]),
                                     all_times < activity.shape[0]-window[1])]
    feat_mat = feature[times]
    # if prev_tr is True, the psth will be computed conditioned on the
    # previous trial values
    if prev_tr:
        values = np.unique(feat_mat)
        rand_feature = np.array(np.random.choice(values, size=(1,)))
        feat_mat = np.concatenate((rand_feature, feat_mat[:-1]))

    act_mat = []
    # get activities
    for ind_t in range(times.shape[0]):
        start = times[ind_t]+window[0]
        end = times[ind_t]+window[1]
        act_mat.append(activity[start:end])

    # bin feature mat
    if feat_bin is not None:
        feat_mat_bin = np.ceil(feat_bin*(feat_mat-np.min(feat_mat)+1e-5) /
                               (np.max(feat_mat)-np.min(feat_mat)+2e-5))
        feat_mat_bin = feat_mat_bin / feat_bin
    else:
        feat_mat_bin = feat_mat
    act_mat = np.array(act_mat)
    # compute average across time if required
    if av_across_time:
        act_mat = np.mean(act_mat, axis=1).reshape((-1, 1))

    # compute averages and significances
    values = np.unique(feat_mat_bin)
    resp_mean = []
    resp_std = []
    significance = []
    for ind_f in range(values.shape[0]):
        index = np.where(feat_mat_bin == values[ind_f])[0]
        feat_resps = act_mat[index, :]
        resp_mean.append(np.mean(feat_resps, axis=0))
        resp_std.append(np.std(feat_resps, axis=0) /
                        np.sqrt(feat_resps.shape[0]))
        for ind_f2 in range(ind_f+1, values.shape[0]):
            feat_resps2 = act_mat[feat_mat_bin == values[ind_f2], :]
            for ind_t in range(feat_resps.shape[1]):
                _, pvalue = sstats.ranksums(feat_resps[:, ind_t],
                                            feat_resps2[:, ind_t])
                significance.append([ind_f, ind_f2, ind_t, pvalue])

    return resp_mean, resp_std, values, significance


def get_psths(states, feature, times, window, index, feat_bin=None,
              pv_th=0.01, sorting=None, av_across_time=False,
              prev_tr=False):
    """
    calls neuron_selectivity for all neurons and sort the averages (stds) by
    percentage of significant values
    """
    means_neurons = []
    stds_neurons = []
    significances = []
    for ind_n in range(states.shape[0]):
        sts_n = states[ind_n, :]
        means, stds, values, sign =\
            neuron_selectivity(sts_n, feature, times, feat_bin=feat_bin,
                               window=window, av_across_time=av_across_time,
                               prev_tr=prev_tr)
        means_neurons.append(means)
        stds_neurons.append(stds)
        if sorting is None:
            sign = np.array(sign)
            perc_sign = 100*np.sum(sign[:, 3] <
                                   pv_th / sign.shape[0]) / sign.shape[0]
            significances.append(perc_sign)

        if ind_n == 0:
            print('values:')
            print(values)
    if sorting is None:
        sorting = np.argsort(-np.array(significances))
    means_neurons = np.array(means_neurons)
    means_neurons = means_neurons[sorting, :, :]
    stds_neurons = np.array(stds_neurons)
    stds_neurons = stds_neurons[sorting, :, :]

    return means_neurons, stds_neurons, values, sorting


def plot_psths(means_mat, stds_mat, values, neurons, index, suptit='',
               trial_int=6.5, dt=100, folder=''):
    """
    plot the psths (averages) returned get_psths
    """
    # this is to plot dashed lines at trials end
    num_tr = dt*means_mat.shape[2]/trial_int
    events = [x*trial_int for x in np.arange(num_tr)
              if x*trial_int < index[-1]]
    f = ut.get_fig(display_mode)
    for ind_n in range(np.min([35, means_mat.shape[0]])):
        means = means_mat[ind_n, :, :]
        stds = stds_mat[ind_n, :, :]
        plt.subplot(7, 5, ind_n+1)
        for ind_plt in range(values.shape[0]):
            if ind_n == 0:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :],
                             label=str(values[ind_plt]))
            else:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            plt.title(str(neurons[ind_n]))
            plot_time_event(events)
        if ind_n != 30:
            plt.xticks([])
        if ind_n == 0:
            f.legend()
    f.suptitle(suptit)
    if folder != '':
        f.savefig(folder + suptit + '.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)


def plot_cond_psths(means_mat1, stds_mat1, means_mat2, stds_mat2, values1,
                    values2, neurons, index, suptit='', trial_int=6.5, dt=100,
                    folder=''):
    """
    plot the psths (averages) returned by get_psths in two different sets:
    rows 2n+1 and 2n. This is basically used for comparison between after
    correct and after error cases.
    """
    # this is to plot dashed lines at trials end
    num_tr = dt*means_mat1.shape[2]/trial_int
    events = [x*trial_int for x in np.arange(num_tr)
              if x*trial_int < index[-1]]
    f = ut.get_fig(display_mode)
    for ind_n in range(np.min([15, means_mat1.shape[0]])):
        means = means_mat1[ind_n, :, :]
        stds = stds_mat1[ind_n, :, :]
        plt.subplot(6, 5, 2*5*np.floor(ind_n/5) + ind_n % 5 + 1)
        for ind_plt in range(values1.shape[0]):
            if ind_n == 0:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :],
                             label=str(values1[ind_plt]))
            else:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            plt.title(str(neurons[ind_n]))
            ax = plt.gca()
            ut.color_axis(ax, color='g')
            plot_time_event(events)
        plt.xticks([])
        if ind_n == 0:
            f.legend()

    for ind_n in range(np.min([15, means_mat1.shape[0]])):
        means = means_mat2[ind_n, :, :]
        stds = stds_mat2[ind_n, :, :]
        plt.subplot(6, 5, 5*(2*np.floor(ind_n/5)+1) + ind_n % 5 + 1)
        for ind_plt in range(values2.shape[0]):
            if ind_n == 0:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :],
                             label=str(values2[ind_plt]))
            else:
                plt.errorbar(index, means[ind_plt, :], stds[ind_plt, :])
            ax = plt.gca()
            ut.color_axis(ax, color='r')
            plot_time_event(events)
        if ind_n != 11:
            plt.xticks([])
            plt.yticks([])
        if ind_n == 0:
            f.legend()
    f.suptitle(suptit)
    if folder != '':
        f.savefig(folder + suptit + '.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)


def mean_neural_activity(file='/home/linux/network_data_492999.npz',
                         fig=False, n_envs=12, env=0, num_steps=100,
                         obs_size=4, num_units=128, window=(0, 1600),
                         part=[[0, 128]], p_lbl=['all'], folder=''):
    """
    plot average activity during a period specified by window. This function
    is actually used to check whether there is any pattern of activity related
    to the different rep/alt blocks.
    """
    states, _, _, _, _, _, _ =\
        get_simulation_vars(file=file, fig=fig, n_envs=n_envs, env=env,
                            num_steps=num_steps, obs_size=obs_size,
                            num_units=num_units)
    dt = 100
    win_l = int(np.diff(window))
    index = np.linspace(dt*window[0], dt*window[1],
                        int(win_l), endpoint=False).reshape((win_l, 1))

    times = np.arange(states.shape[1]//window[1])*window[1]
    trial_int = np.mean(np.diff(times))*dt
    feature = np.zeros((states.shape[1],))
    for ind_p in range(len(part)):
        sort = np.arange(np.diff(part[ind_p])[0])
        sts_part = states[part[ind_p][0]:part[ind_p][1], :]
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 feature,
                                                                 times, window,
                                                                 index,
                                                                 sorting=sort)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='mean_act_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)


def neural_analysis(file='/home/linux/network_data_492999.npz',
                    fig=False, n_envs=12, env=0, num_steps=100,
                    obs_size=4, num_units=128, window=(-5, 20),
                    part=[[0, 128]], p_lbl=['all'], folder=''):
    """
    get variables from experiment in file and plot selectivities to:
    action, reward, stimulus and action conditioned on prev. reward
    """
    states, rewards, actions, obs, trials, _, _ =\
        get_simulation_vars(file=file, fig=fig, n_envs=n_envs, env=env,
                            num_steps=num_steps, obs_size=obs_size,
                            num_units=num_units)
    dt = 100
    win_l = int(np.diff(window))
    index = np.linspace(dt*window[0], dt*window[1],
                        int(win_l), endpoint=False).reshape((win_l, 1))
    times = np.where(trials == 1)[0]
    trial_int = np.mean(np.diff(times))*dt
    for ind_p in range(len(part)):
        print(p_lbl[ind_p])
        sts_part = states[part[ind_p][0]:part[ind_p][1], :]
        # actions
        print('selectivity to actions')
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 actions,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='act_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)

        # rewards
        print('selectivity to reward')
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 rewards,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='rew_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)

        # obs
        print('selectivity to cumulative observation')
        obs_cum = np.zeros_like(obs)
        for ind_t in range(times.shape[0]):
            if ind_t == 0:
                obs_cum[times[ind_t]] = np.sum(obs[0: times[ind_t]])
            else:
                obs_cum[times[ind_t]] = np.sum(obs[times[ind_t-1]:
                                                   times[ind_t]])
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 obs_cum,
                                                                 times,
                                                                 window,
                                                                 index,
                                                                 feat_bin=4)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='cum_obs_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)

        print('selectivity to action conditioned on reward')
        times_r = times[np.where(rewards[times] == 1)]
        means_r, stds_r, values_r, sorting = get_psths(sts_part, actions,
                                                       times_r, window, index)

        times_nr = times[np.where(rewards[times] == 0)]
        means_nr, stds_nr, values_nr, _ = get_psths(sts_part, actions,
                                                    times_nr,
                                                    window, index,
                                                    sorting=sorting)
        plot_cond_psths(means_r, stds_r, means_nr, stds_nr,
                        values_r, values_nr, sorting, index,
                        suptit='act_cond_rew_select_' + p_lbl[ind_p],
                        trial_int=trial_int, dt=dt, folder=folder)

        print('selectivity to actions cond on low stimulus ev')
        times_lowEv =\
            times[np.where(np.abs(obs_cum[times]) <
                           np.percentile(np.abs(obs_cum[times]), 10))]
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 actions,
                                                                 times_lowEv,
                                                                 window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='act_cond_lowStim_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)

        print('selectivity to actions cond on high stimulus ev')
        times_lowEv =\
            times[np.where(np.abs(obs_cum[times]) >
                           np.percentile(np.abs(obs_cum[times]), 90))]
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 actions,
                                                                 times_lowEv,
                                                                 window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='act_cond_highStim_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)


def transition_analysis(file='/home/linux/network_data_492999.npz',
                        fig=False, n_envs=12, env=0, num_steps=100,
                        obs_size=4, num_units=128, window=(-5, 20),
                        part=[[0, 128]], p_lbl=['all'], folder=''):
    """
    get variables from experiment in file and plot selectivities to
    transition evidence
    """
    dt = 100
    win_l = int(np.diff(window))
    index = np.linspace(dt*window[0], dt*window[1],
                        int(win_l), endpoint=False).reshape((win_l, 1))
    states, rewards, actions, obs, trials, _, _ =\
        get_simulation_vars(file=file, fig=fig, n_envs=n_envs, env=env,
                            num_steps=num_steps, obs_size=obs_size,
                            num_units=num_units)
    times = np.where(trials == 1)[0]
    trial_int = np.mean(np.diff(times))*dt
    choice = actions[times]
    outcome = rewards[times]
    ground_truth = choice.copy()
    ground_truth[np.where(ground_truth == 2)] = -1
    ground_truth *= (-1)**(outcome == 0)
    num_steps = trials.shape[0]
    trans_mat = get_transition_mat(ground_truth, times, num_steps=num_steps,
                                   conv_window=4)
    for ind_p in range(len(part)):
        sts_part = states[part[ind_p][0]:part[ind_p][1], :]
        print('selectivity to number of repetitions')
        means_neurons, stds_neurons, values, sorting = get_psths(sts_part,
                                                                 trans_mat,
                                                                 times, window,
                                                                 index)
        plot_psths(means_neurons, stds_neurons, values, sorting, index,
                   suptit='num_rep_select_' + p_lbl[ind_p],
                   trial_int=trial_int, dt=dt, folder=folder)

        print('selectivity to num of repetitions conditioned on prev. reward')
        rews = np.where(rewards[times] == 1)[0]+1
        times_prev_r = times[rews[:-1]]
        means_r, stds_r, values_r, sorting = get_psths(sts_part, trans_mat,
                                                       times_prev_r, window,
                                                       index)
        non_rews = np.where(rewards[times] == 0)[0]+1
        times_prev_nr = times[non_rews[:-1]]
        means_nr, stds_nr, values_nr, _ = get_psths(sts_part, trans_mat,
                                                    times_prev_nr, window,
                                                    index, sorting=sorting)
        plot_cond_psths(means_r, stds_r, means_nr, stds_nr,
                        values_r, values_nr, sorting, index,
                        suptit='num_rep_cond_prev_rew_select_' + p_lbl[ind_p],
                        trial_int=trial_int, dt=dt, folder=folder)


def bias_analysis(file='/home/linux/network_data_492999.npz',
                  fig=False, n_envs=12, env=0, num_steps=100,
                  obs_size=4, num_units=128, window=(-5, 20), folder=''):
    """
    get variables from experiment in file and plot selectivities to
    bias (transition evidence x previous choice)
    """
    dt = 100
    win_l = int(np.diff(window))
    index = np.linspace(dt*window[0], dt*window[1],
                        int(win_l), endpoint=False).reshape((win_l, 1))
    states, rewards, actions, obs, trials, _, _ =\
        get_simulation_vars(file=file, fig=fig, n_envs=n_envs, env=env,
                            num_steps=num_steps, obs_size=obs_size,
                            num_units=num_units)
    times = np.where(trials == 1)[0]
    trial_int = np.mean(np.diff(times))*dt
    choice = actions[times]
    outcome = rewards[times]
    ground_truth = choice.copy()
    ground_truth[np.where(ground_truth == 2)] = -1
    ground_truth *= (-1)**(outcome == 0)
    num_steps = trials.shape[0]
    trans_mat = get_transition_mat(ground_truth, times, num_steps=num_steps,
                                   conv_window=4)
    rand_choice = np.array(np.random.choice([1, 2])).reshape(1,)
    previous_choice = np.concatenate((rand_choice, choice[:-1]))
    previous_choice[np.where(previous_choice == 2)] = -1
    bias_mat = trans_mat.copy()
    for ind_t in range(times.shape[0]):
        bias_mat[times[ind_t]] *= previous_choice[ind_t]
    print('selectivity to bias')
    means_neurons, stds_neurons, values, sorting = get_psths(states,
                                                             bias_mat,
                                                             times, window,
                                                             index)
    plot_psths(means_neurons, stds_neurons, values, sorting, index,
               suptit='biasd_select', trial_int=trial_int, dt=dt,
               folder=folder)
    print('selectivity to bias conditioned on reward')
    window = (-2, 0)
    win_l = int(np.diff(window))
    index = np.linspace(dt*window[0], dt*window[1],
                        int(win_l), endpoint=False).reshape((win_l, 1))
    rews = np.where(rewards[times] == 1)[0]+1
    times_prev_r = times[rews[:-1]]
    means_r, stds_r, values_r, sorting = get_psths(states, bias_mat,
                                                   times_prev_r, window,
                                                   index,
                                                   av_across_time=True)
    non_rews = np.where(rewards[times] == 0)[0]+1
    times_prev_nr = times[non_rews[:-1]]
    means_nr, stds_nr, values_nr, _ = get_psths(states, bias_mat,
                                                times_prev_nr, window,
                                                index, sorting=sorting,
                                                av_across_time=True)
    assert (values_r == values_nr).all()
    f = ut.get_fig(display_mode)
    rows = 9
    cols = 7
    for ind_n in range(int(rows*cols)):
        corr_r = np.abs(np.corrcoef(values_r,
                                    means_r[ind_n, :, :].T))[0, 1]
        corr_nr = np.abs(np.corrcoef(values_r,
                                     means_nr[ind_n, :, :].T))[0, 1]
        plt.subplot(rows, cols, ind_n+1)
        plt.errorbar(values_r, means_r[ind_n, :, :], stds_r[ind_n, :, :],
                     label='after correct')
        plt.errorbar(values_r, means_nr[ind_n, :, :], stds_nr[ind_n, :, :],
                     label='after error')
        plt.title(str(np.round(corr_r, 2)) +
                  ' | ' + str(np.round(corr_nr, 2)))
    plt.legend()
    f.suptitle('selectivity to bias conditionate on previous reward')
    if folder != '':
        f.savefig(folder + 'bias_cond_prev_rew_select' + '.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)
    print('firing rate correlation with bias before stimulus')
    f = ut.get_fig(display_mode)
    for ind_n in range(means_r.shape[0]):
        corr_r = np.abs(np.corrcoef(values_r,
                                    means_r[ind_n, :, :].T))[0, 1]
        corr_nr = np.abs(np.corrcoef(values_r,
                                     means_nr[ind_n, :, :].T))[0, 1]
        plt.plot(corr_r, corr_nr, '.', markerSize=2)
    if np.isnan(corr_r):
        corr_r = 0
    if np.isnan(corr_nr):
        corr_nr = 0
    plt.xlabel('correlation after correct')
    plt.ylabel('correlation after error')
    plt.title('correlation between baseline firing rate and bias')
    if folder != '':
        f.savefig(folder + 'activity_bias_corr' + '.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)


############################################
# BEHAVIOR ANALYSIS
############################################


def load_behavioral_data(file):
    """
    loads behavioral data and get relevant info from it
    """
    data = np.load(file)
    rep_prob = data['rep_prob']
    choice = data['choice']
    stimulus = data['stimulus']
    correct_side = data['correct_side']
    performance = (choice == correct_side)
    evidence = stimulus[:, 1] - stimulus[:, 2]
    return choice, correct_side, performance, evidence, rep_prob


def plot_learning(performance, evidence, stim_position, w_conv=200,
                  legend=False):
    """
    plots RNN and ideal observer performances.
    The function assumes that a figure has been created
    before it is called.
    """
    num_trials = performance.shape[0]
    # remove all previous plots
    # ideal observer choice
    io_choice = (evidence < 0) + 1
    io_performance = io_choice == stim_position
    # save the mean performances
    RNN_perf = np.mean(performance[2000:].flatten())
    io_perf = np.mean(io_performance[2000:].flatten())

    # plot smoothed performance
    performance_smoothed = np.convolve(performance,
                                       np.ones((w_conv,))/w_conv,
                                       mode='valid')
    performance_smoothed = performance_smoothed[0::w_conv]
    plt.plot(np.linspace(0, num_trials, performance_smoothed.shape[0]),
             performance_smoothed, color=(0.39, 0.39, 0.39), lw=0.5,
             label='RNN perf. (' + str(round(RNN_perf, 3)) + ')')
    print('RNN perf: ' + str(round(RNN_perf, 3)))
    # plot ideal observer performance
    io_perf_smoothed = np.convolve(io_performance,
                                   np.ones((w_conv,))/w_conv,
                                   mode='valid')
    io_perf_smoothed = io_perf_smoothed[0::w_conv]
    plt.plot(np.linspace(0, num_trials, io_perf_smoothed.shape[0]),
             io_perf_smoothed, color=(1, 0.8, 0.5), lw=0.5,
             label='Ideal Obs. perf. (' + str(round(io_perf, 3)) + ')')
    # plot 0.25, 0.5 and 0.75 performance lines
    plot_fractions([0, performance.shape[0]])
    plt.title('performance')
    plt.xlabel('trials')
    if legend:
        plt.legend()


def bias_across_training(choice, evidence, performance,
                         per=100000, conv_window=3):
    """
    compute bias across training
    """
    num_stps = int(choice.shape[0] / per)
    transitions = get_transition_mat(choice, conv_window=conv_window)
    perf_hist = np.convolve(performance, np.ones((conv_window,)),
                            mode='full')[0:-conv_window+1]
    perf_hist = np.concatenate((np.array([0]), perf_hist[:-1]))
    values = np.unique(transitions)
    bias_mat = []
    for ind_per in range(num_stps):
        ev = evidence[ind_per*per:(ind_per+1)*per]
        perf = performance[ind_per*per:(ind_per+1)*per]
        ch = choice[ind_per*per:(ind_per+1)*per]
        trans = transitions[ind_per*per:(ind_per+1)*per]
        p_hist = perf_hist[ind_per*per:(ind_per+1)*per]
        for ind_perf in range(2):
            for ind_tr in [0, values.shape[0]-1]:
                mask = np.logical_and.reduce((trans == values[ind_tr],
                                              perf == ind_perf,
                                              p_hist == conv_window))
                mask = np.concatenate((np.array([False]), mask[:-1]))
                if ind_perf == 0 and ind_tr == 0 and\
                   ind_per == num_stps-1 and False:
                    # repeat = get_repetitions(ch)
                    ut.get_fig()
                    num = 50
                    start = 5000
                    #                    plt.plot(ch[start:start+num], '-+',
                    #                             label='choice', lw=1)
                    plt.plot(trans[start:start+num], '-+',
                             label='transitions', lw=1)
                    plt.plot(perf[start:start+num]-3, '--+', label='perf',
                             lw=1)
                    plt.plot(mask[start:start+num]-3, '-+', label='mask',
                             lw=1)
                    # plt.plot(repeat[start:start+num]*2, '-+', label='repeat',
                    # lw=1)
                    plt.plot(p_hist[start:start+num], '-+',
                             label='perf_hist', lw=1)
                    for ind in range(num):
                        plt.plot([ind, ind], [-3, 3], '--',
                                 color=(.7, .7, .7))
                    plt.legend()

                if np.sum(mask) > 100:
                    popt, pcov = bias_calculation(ch.copy(), ev.copy(),
                                                  mask.copy())
                else:
                    popt = [0, 0]
                bias_mat.append([popt[1], ind_perf,
                                 ind_tr/(values.shape[0]-1)])
    bias_mat = np.array(bias_mat)
    return bias_mat


def plot_bias_across_training(bias_mat, tot_num_trials, folder='',
                              fig=True, legend=False, per=100000,
                              conv_window=3):
    """
    plot the results obtained by bias_across_training
    """
    num_stps = int(tot_num_trials / per)
    time_stps = np.linspace(per, tot_num_trials, num_stps)
    lbl_perf = ['error', 'correct']
    lbl_trans = ['alts', 'reps']
    if fig:
        f = ut.get_fig(display_mode)
    for ind_perf in range(2):
        for ind_tr in range(2):
            if ind_perf == 0:
                color = np.array([1-0.25*ind_tr, 0.75, 0.75*(ind_tr + 1)])
                color[color > 1] = 1
            else:
                color = ((1-ind_tr), 0, ind_tr)
            index = np.logical_and(bias_mat[:, 1] == ind_perf,
                                   bias_mat[:, 2] == ind_tr)
            plt.plot(time_stps, bias_mat[index, 0], '+-', color=color, lw=1,
                     label=str(conv_window) + ' ' + lbl_trans[ind_tr] +
                     ' after ' + lbl_perf[ind_perf])
    plt.title('bias after ' + str(conv_window) +
              ' correct trans. across training')
    if legend:
        plt.legend()
    if folder != '' and fig:
        f.savefig(folder + 'bias_evolution.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)


def perf_cond_on_stim_ev(file='/home/linux/PassAction.npz', save_path='',
                         fig=True):
    """
    computes performance as a function of the stimulus evidence
    """
    _, _, performance, evidence, _ = load_behavioral_data(file)
    evidence = evidence[-2000000:]
    performance = performance[-2000000:]
    perf_mat = []
    for ind_ev in range(10):
        mask_ev = np.logical_and(evidence >= np.percentile(evidence,
                                                           ind_ev*10),
                                 evidence <= np.percentile(evidence,
                                                           (ind_ev+1)*10))
        perf_mat.append(np.mean(performance[mask_ev].flatten()))
    ut.get_fig()
    plt.plot(np.arange(10)*10+5, perf_mat, '-+')
    plt.xlabel('stim evidence percentile')
    plt.ylabel('performance')
    print('Mean performance: ' + str(np.mean(performance)))
    ut.get_fig()
    sh_mat = []
    # num_bins = 20
    index = np.arange(50)
    for ind_sh in range(10):
        shuffled = performance.copy()
        np.random.shuffle(shuffled)
        inter_error_distance = np.diff(np.where(shuffled == 0)[0])
        assert (inter_error_distance != 0).all()
        sh_hist = np.histogram(inter_error_distance, index)[0]
        sh_hist = sh_hist / np.sum(sh_hist)
        sh_mat.append(sh_hist)

    sh_mat = np.array(sh_mat)
    or_hist = np.histogram(np.diff(np.where(performance == 0)[0]), index)[0]
    or_hist = or_hist / np.sum(or_hist)
    plt.errorbar(index[:-1], np.mean(sh_mat, axis=0), np.std(sh_mat, axis=0),
                 label='shuffled')
    for ind_p in np.arange(10)*0.1:
        sh_mat2 = np.random.binomial(1, ind_p, size=performance.shape)
        sh_hist = np.histogram(np.diff(np.where(sh_mat2 == 0)[0]), index)[0]
        sh_hist = sh_hist / np.sum(sh_hist)
        plt.plot(index[:-1], sh_hist, '--', color=(.9, .9, .9),
                 label='binomial p=' + str(ind_p))
    plt.plot(index[:-1], or_hist, label='orignal')
    plt.legend()
    plt.xlabel('distance between errors')
    plt.ylabel('count')


def bias_after_altRep_seqs(file='/home/linux/PassReward0_data.npz',
                           num_tr=100000):
    """
    computes bias conditioned on the num. of previous consecutive ground truth
    alternations/repetitions for after correct/error trials
    """
    choice, correct_side, performance, evidence, _ = load_behavioral_data(file)
    # BIAS CONDITIONED ON TRANSITION HISTORY (NUMBER OF REPETITIONS)
    start_point = performance.shape[0]-num_tr
    ev = evidence[start_point:start_point+num_tr]
    perf = performance[start_point:start_point+num_tr]
    ch = choice[start_point:start_point+num_tr]
    side = correct_side[start_point:start_point+num_tr]
    mat_biases = []
    mat_conv = np.arange(1, num_trials_back)
    mat_num_samples = np.zeros((num_trials_back-1, 2))
    for conv_window in mat_conv:
        # get number of repetitions during the last conv_window trials
        # (not including the current trial)
        if conv_window > 1:
            transitions = get_transition_mat(ch, conv_window=conv_window)
        else:
            repeat = get_repetitions(ch)
            transitions = np.concatenate((np.array([0]), repeat[:-1]))
        if conv_window == 2:
            transitions_side = get_transition_mat(side,
                                                  conv_window=conv_window)
        # perf_hist is use to check that all previous last trials where correct
        if conv_window > 1:
            perf_hist = np.convolve(perf, np.ones((conv_window,)),
                                    mode='full')[0:-conv_window+1]
            perf_hist = np.concatenate((np.array([0]), perf_hist[:-1]))
        else:
            perf_hist = np.concatenate((np.array([0]), perf[:-1]))
        values = np.unique(transitions)
        for ind_perf in range(2):
            for ind_tr in [0, values.shape[0]-1]:
                # mask finds all times in which the current trial is
                # correct/error and the trial history (num. of repetitions)
                # is values[ind_tr] we then need to shift these times
                # to get the bias in the trial following them
                mask = np.logical_and.reduce((transitions == values[ind_tr],
                                              perf == ind_perf,
                                              perf_hist == conv_window))
                mask = np.concatenate((np.array([False]), mask[:-1]))
                if conv_window == 2:
                    aux = transitions_side == values[ind_tr]
                    aux2 = perf_hist == conv_window
                    mask_side = np.logical_and.reduce((aux,
                                                       perf == ind_perf,
                                                       aux2))
                    mask_side = np.concatenate((np.array([False]),
                                                mask_side[:-1]))
                    index = np.where(mask_side != mask)[0]
                if conv_window == 2 and False:
                    num = 50
                    start = index[0]-10
                    plt.figure()
                    for ind in range(num):
                        plt.plot([ind, ind], [-2, 2], '--', color=(.6, .6, .6))
                    repeat = get_repetitions(ch)
                    plt.plot(repeat[start:start+num]-1, '-+', lw=1,
                             label='repeat')
                    plt.plot(transitions[start:start+num], '-+', lw=1,
                             label='transitions')
                    plt.plot(transitions_side[start:start+num], '--+', lw=1,
                             label='transitions side')
                    plt.plot(perf_hist[start:start+num], '-+', lw=1,
                             label='perf_hist')
                    plt.plot(perf[start:start+num]-3, '-+', lw=1,
                             label='performance')
                    plt.plot(mask[start:start+num]-3, '-+', lw=1,
                             label='mask')
                    plt.plot(mask_side[start:start+num]-3, '--+', lw=1,
                             label='mask_ch')
                    plt.legend()

                mat_num_samples[conv_window-1, ind_perf] += np.sum(mask)
                if np.sum(mask) > 100:
                    popt, _ = bias_calculation(ch.copy(), ev.copy(),
                                               mask.copy())
                else:
                    popt = [0, 0]
                # here I want to compute the bias at t+2 later when the trial
                # 1 step later was correct
                next_perf = np.concatenate((np.array([0]), perf[:-1]))
                mask = np.logical_and.reduce((transitions == values[ind_tr],
                                              perf == ind_perf,
                                              perf_hist == conv_window,
                                              next_perf == 1))

                mask = np.concatenate((np.array([False, False]), mask[:-2]))
                if np.sum(mask) > 100:
                    popt_next, _ = bias_calculation(ch.copy(), ev.copy(),
                                                    mask.copy())
                else:
                    popt_next = [0, 0]
                mat_biases.append([popt[1], ind_perf,
                                   ind_tr/(values.shape[0]-1), conv_window,
                                   popt_next[1]])
    mat_biases = np.array(mat_biases)
    return mat_biases, mat_num_samples


def plot_bias_after_altRep_seqs(mat_biases, mat_conv, mat_num_samples,
                                folder='', panels=None, legend=False):
    lbl_perf = ['error', 'correct']
    lbl_tr = ['alt', 'rep']
    if panels is None:
        f = ut.get_fig(display_mode)
    for ind_perf in range(2):
        if panels is None:
            plt.subplot(1, 2, int(not(ind_perf))+1)
        else:
            plt.subplot(panels[0], panels[1], panels[2]+int(not(ind_perf)))
        ax = plt.gca()
        ax.set_title('bias after ' + lbl_perf[ind_perf])
        # plot number of samples per number of rep/alt transition
        ax2 = ax.twinx()
        ax2.plot(mat_conv, mat_num_samples[:, ind_perf], color=(.7, .7, .7))
        ax2.set_ylabel('total num. of samples')
        for ind_tr in range(2):
            color = ((1-ind_tr), 0, ind_tr)
            index = np.logical_and(mat_biases[:, 1] == ind_perf,
                                   mat_biases[:, 2] == ind_tr)
            ax.plot(mat_conv, mat_biases[index, 0], '+-', color=color, lw=1,
                    label=lbl_tr[ind_tr] + ' + ' + lbl_perf[ind_perf])
            ax.plot(mat_conv, mat_biases[index, 4], '--', color=color, lw=1,
                    label=lbl_tr[ind_tr] + ' at t+2')
        if legend:
            plt.legend()
        ax.set_ylabel('bias')
        ax.set_xlabel('number of ground truth transitions')
    if (panels is None) and folder != '':
        f.savefig(folder + 'bias_after_saltRep_seqs.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)
    return mat_biases


def bias_after_transEv_change(file='/home/linux/PassReward0_data.npz',
                              num_tr=100000):
    """
    computes bias conditioned on the number of consecutive ground truth
    alternations/repetitions during the last trials
    """
    choice, correct_side, performance, evidence, _ = load_behavioral_data(file)
    # BIAS CONDITIONED ON TRANSITION HISTORY (NUMBER OF REPETITIONS)
    start_point = performance.shape[0]-num_tr
    ev = evidence[start_point:start_point+num_tr]
    ch = choice[start_point:start_point+num_tr]
    side = correct_side[start_point:start_point+num_tr]
    perf = performance[start_point:start_point+num_tr]
    next_perf = np.concatenate((np.array([0]), perf[:-1]))
    mat_biases = []
    mat_conv = np.arange(1, num_trials_back)
    for conv_window in mat_conv:
        # get number of repetitions during the last conv_window trials
        # (not including the current trial)
        # get number of repetitions during the last conv_window trials
        # (not including the current trial)
        if conv_window > 1:
            trans_gt = get_transition_mat(side, conv_window=conv_window)
        else:
            repeat = get_repetitions(side)
            trans_gt = np.concatenate((np.array([0]), repeat[:-1]))
        # use only extreme cases (all alt., all  rep.)
        abs_trans = np.abs(trans_gt)
        # the order of the concatenation is imposed by the fact that
        # transitions measures the trans. ev. in the previous trials, *not
        # counting the current trial*
        tr_change = np.concatenate((abs_trans, abs_trans[0].reshape((1,))))
        tr_change = np.diff(tr_change)
        values = np.unique(tr_change)
        # now get choice transitions
        if conv_window > 1:
            trans = get_transition_mat(ch, conv_window=conv_window)
        else:
            repeat = get_repetitions(ch)
            trans = np.concatenate((np.array([0]), repeat[:-1]))
        values_tr = np.unique(trans)
        # perf_hist is use to check that all previous last trials where correct
        if conv_window > 1:
            perf_hist = np.convolve(perf, np.ones((conv_window,)),
                                    mode='full')[0:-conv_window+1]
            perf_hist = np.concatenate((np.array([0]), perf_hist[:-1]))
        else:
            perf_hist = np.concatenate((np.array([0]), perf[:-1]))
        for ind_tr in [0, values_tr.shape[0]-1]:
            # since we are just looking at the extrem cases (see above),
            # there cannot be an increase in transition evidence
            for ind_ch in range(2):
                for ind_perf in range(2):
                    # mask finds all times in which the current trial is
                    # correct/error and the trial history (num. of repetitions)
                    # is values[ind_tr] we then need to shift these times
                    # to get the bias in the trial following them
                    mask = np.logical_and.reduce((tr_change == values[ind_ch],
                                                 next_perf == ind_perf,
                                                 trans == values_tr[ind_tr],
                                                 perf_hist == conv_window))
                    mask = np.concatenate((np.array([False]), mask[:-1]))
                    if conv_window == 2 and ind_ch == 0 and False:
                        repeat_choice = get_repetitions(ch)
                        print(np.where(mask == 1))
                        print('tr change : ' + str(values[ind_ch]))
                        print('perf. : ' + str(ind_perf))
                        print('transition : ' + str(values_tr[ind_tr]))
                        print('conv. : ' + str(conv_window))
                        ut.get_fig()
                        num = 700
                        start = 160
                        plt.title('change:' + str(values[ind_ch]) +
                                  '  perf:' + str(ind_perf) +
                                  '  trans:' + str(values_tr[ind_tr]))
                        plt.plot(trans_gt[start:start+num], '-+',
                                 label='transitions gt', lw=1)
                        plt.plot(tr_change[start:start+num], '-+',
                                 label='tr_change', lw=1)
                        plt.plot(perf[start:start+num]-3, '--+', label='perf',
                                 lw=1)
                        plt.plot(mask[start:start+num]-3, '-+', label='mask',
                                 lw=1)
                        plt.plot(repeat_choice[start:start+num]+2, '-+',
                                 label='repeat', lw=1)
                        plt.plot(trans[start:start+num], '--+',
                                 label='transitions choice', lw=1)
                        for ind in range(num):
                            plt.plot([ind, ind], [-3, 3], '--',
                                     color=(.7, .7, .7))
                        plt.legend()
                        # asdas
                    if np.sum(mask) > 100:
                        popt, pcov = bias_calculation(ch.copy(), ev.copy(),
                                                      mask.copy())
                    else:
                        popt = [0, 0]
                    mat_biases.append([popt[1], ind_ch, ind_perf,
                                       ind_tr/(values_tr.shape[0]-1),
                                       conv_window, np.sum(mask)])
    mat_biases = np.array(mat_biases)
    return mat_biases


def plot_bias_after_transEv_change(mat_biases, folder, panels=None,
                                   legend=False):
    if panels is None:
        f = ut.get_fig(display_mode)
    lbl_ch = ['less', 'equal']  # , 'more evidence']
    lbl_perf = ['error', 'correct']
    lbl_tr = ['alt. bl.', 'rep. bl.']

    for ind_ch in range(2):
        if panels is None:
            plt.subplot(1, 2, ind_ch+1)
            plt.title('after change to ' +
                      lbl_ch[ind_ch] + ' transition evidence')
        else:
            plt.subplot(panels[0], panels[1], panels[2]+ind_ch)
            plt.title('after change to ' +
                      lbl_ch[ind_ch] + ' transition evidence')
        for ind_tr in range(2):
            for ind_perf in range(2):
                if ind_perf == 0:
                    color = np.array([1-0.25*ind_tr, 0.75,
                                      0.75*(ind_tr + 1)])
                    color[color > 1] = 1
                else:
                    color = ((1-ind_tr), 0, ind_tr)
                    index = np.logical_and.reduce((mat_biases[:, 1] == ind_ch,
                                                  mat_biases[:, 2] == ind_perf,
                                                  mat_biases[:, 3] == ind_tr))
                    label = lbl_tr[ind_tr] + ', after ' + lbl_perf[ind_perf]
                    plt.plot(mat_biases[index, 4], mat_biases[index, 0],
                             color=color, lw=1, label=label)
        if legend:
            plt.legend()
        plt.ylabel('bias')
        plt.xlabel('number of ground truth transitions')

    if (panels is None) and folder != '':
        f.savefig(folder + 'bias_after_trans_ev_change.png',
                  dpi=DPI, bbox_inches='tight')
        plt.close(f)


def batch_analysis(main_folder, neural_analysis_flag=False,
                   behavior_analysis_flag=True):
    per = 100000
    saving_folder_all = main_folder + 'all_results/'
    if not os.path.exists(saving_folder_all):
        os.mkdir(saving_folder_all)
    experiments, expl_params = res_summ.explore_folder(main_folder,
                                                       count=False)
    expl_keys = list(expl_params.keys())
    inter_exp_biases = []
    for exp in experiments:
        params = exp[0]
        print('------------------------')
        p_exp = {k: params[k] for k in params if k in expl_params}
        print(p_exp)
        if params['network'] == 'twin_net':
            params['nlstm'] *= 2
        _, folder = cf.build_command(save_folder=main_folder,
                                     ps_r=params['pass_reward'],
                                     ps_act=params['pass_action'],
                                     bl_dur=params['bl_dur'],
                                     num_u=params['nlstm'],
                                     nsteps=params['nsteps'],
                                     stimEv=params['stimEv'],
                                     net_type=params['network'],
                                     num_stps_env=params['num_timesteps'],
                                     save=False, alg=params['alg'],
                                     rep_prob=params['rep_prob'])
        folder = os.path.basename(os.path.normpath(folder + '/'))
        # only late experiments indicate the parameter alpha explicitly in the
        # name of the folder
        if 'alpha' not in expl_keys:
            undscr_ind = folder.rfind('_a_')
            folder_name = folder[:undscr_ind]
            files = glob.glob(main_folder + folder_name + '*')
        else:
            undscr_ind = folder.rfind('_')
            folder_name = folder[:undscr_ind]
            print(folder_name)
            files = glob.glob(main_folder + folder_name + '*')

        if len(files) > 0:
            f = ut.get_fig(display_mode)
            biases_after_seqs = []
            biases_after_transEv = []
            bias_acr_training = []
            num_samples_mat = []
            performances = []
            for ind_f in range(len(files)):
                file = files[ind_f] + '/bhvr_data_all.npz'
                data_flag = ptf.put_files_together(files[ind_f],
                                                   min_num_trials=per)
                if data_flag:
                    performances, bias_acr_training, biases_after_seqs,\
                        biases_after_transEv, num_samples_mat =\
                        get_main_results(file, bias_acr_training,
                                         biases_after_seqs,
                                         biases_after_transEv, num_samples_mat,
                                         per, performances)
                    plot_main_results(file, bias_acr_training,
                                      biases_after_seqs,
                                      biases_after_transEv,
                                      num_samples_mat, ind_f == 0, per)
            set_yaxis()
            # get biases
            biases, biases_t_2 = organize_biases(biases_after_seqs)
            p_exp['biases'] = biases
            p_exp['perfs'] = performances
            p_exp['num_exps'] = biases.shape[0]
            inter_exp_biases.append(p_exp)
            results = {'biases_after_transEv': biases_after_transEv,
                       'biases_after_seqs': biases_after_seqs,
                       'bias_across_training': bias_acr_training,
                       'num_samples_mat': num_samples_mat,
                       'biases': biases,
                       'biases_t_2': biases_t_2,
                       'performances': performances}
            np.savez(saving_folder_all + '/' + folder_name +
                     '_results.npz', **results)
            f.savefig(saving_folder_all + '/' + folder_name +
                      '_bhvr_fig.png', dpi=DPI, bbox_inches='tight')
        plot_biases_all_experiments(inter_exp_biases, expl_params,
                                    saving_folder_all)
        plot_bias_ratios_all_experiments(inter_exp_biases, expl_params,
                                         saving_folder_all)
        plot_perf_all_experiments(inter_exp_biases, expl_params,
                                  saving_folder_all)


def plot_biases_all_experiments(inter_exp_biases, expl_params,
                                saving_folder_all):
    f_bias = ut.get_fig(display_mode)
    inter_exp_biases = sort_results(inter_exp_biases, expl_params)
    xticks = []
    for ind_exp in range(len(inter_exp_biases)):
        p_exp = inter_exp_biases[ind_exp].copy()
        all_biases = p_exp['biases']
        mat_means = np.mean(all_biases, axis=0)
        mat_std = np.std(all_biases, axis=0)
        num_exps = p_exp['num_exps']
        del p_exp['biases']
        del p_exp['perfs']
        print(p_exp)
        specs = json.dumps(p_exp)
        specs = reduce_xticks(specs)
        xticks.append(specs)
        counter = 0
        for ind_perf in range(2):
            for ind_tr in range(2):
                color = np.array(((1-ind_tr), 0, ind_tr)) + 0.5*(1-ind_perf)
                color[color > 1] = 1
                mean = mat_means[2, counter]
                std = mat_std[2, counter]
                plt.plot(np.random.normal(loc=ind_exp,
                                          scale=0.01*len(inter_exp_biases),
                                          size=(all_biases.shape[0],)),
                         all_biases[:, 2, counter], '.', color=color,
                         markerSize=5, alpha=1.)
                plt.errorbar(ind_exp, mean, std/np.sqrt(num_exps),
                             marker='+', color=color, markerSize=10)
                counter += 1
    plt.xticks(np.arange(len(inter_exp_biases)), xticks)
    plt.xlim([-0.5, len(inter_exp_biases)-0.5])
    f_bias.savefig(saving_folder_all + '/all_together_bias.png', dpi=DPI,
                   bbox_inches='tight')


def plot_bias_ratios_all_experiments(inter_exp_biases, expl_params,
                                     saving_folder_all):
    f_bias = ut.get_fig(display_mode)
    inter_exp_biases = sort_results(inter_exp_biases, expl_params)
    xticks = []
    for ind_exp in range(len(inter_exp_biases)):
        p_exp = inter_exp_biases[ind_exp].copy()
        all_biases = p_exp['biases']
        num_exps = p_exp['num_exps']
        del p_exp['biases']
        del p_exp['perfs']
        print(p_exp)
        specs = json.dumps(p_exp)
        specs = reduce_xticks(specs)
        xticks.append(specs)
        counter = 0
        for ind_tr in range(2):
            color = np.array(((1-ind_tr), 0, ind_tr))
            color[color > 1] = 1
            ratios = all_biases[:, 2, ind_tr]/all_biases[:, 2, 2+ind_tr]
            mean = np.mean(ratios)
            std = np.std(ratios)
            plt.plot(np.random.normal(loc=ind_exp,
                                      scale=0.01*len(inter_exp_biases),
                                      size=(all_biases.shape[0],)),
                     ratios, '.', color=color,
                     markerSize=5, alpha=1.)
            plt.errorbar(ind_exp, mean, std/np.sqrt(num_exps),
                         marker='+', color=color, markerSize=10)
            counter += 1
    plt.xticks(np.arange(len(inter_exp_biases)), xticks)
    plt.xlim([-0.5, len(inter_exp_biases)-0.5])
    f_bias.savefig(saving_folder_all + '/all_together_ratios.png', dpi=DPI,
                   bbox_inches='tight')


def plot_perf_all_experiments(inter_exp_biases, expl_params,
                              saving_folder_all):
    f_bias = ut.get_fig(display_mode)
    inter_exp_biases = sort_results(inter_exp_biases, expl_params)
    xticks = []
    for ind_exp in range(len(inter_exp_biases)):
        p_exp = inter_exp_biases[ind_exp].copy()
        all_perfs = p_exp['perfs']
        mat_means = np.mean(all_perfs, axis=0)
        mat_std = np.std(all_perfs, axis=0)
        num_exps = p_exp['num_exps']
        del p_exp['biases']
        del p_exp['perfs']
        specs = json.dumps(p_exp)
        specs = reduce_xticks(specs)
        xticks.append(specs)
        plt.plot(np.random.normal(loc=ind_exp,
                                  scale=0.01*len(inter_exp_biases),
                                  size=(len(all_perfs),)), all_perfs, '.',
                 color='b', markerSize=3, alpha=1.)
        plt.errorbar(ind_exp, mat_means, mat_std/np.sqrt(num_exps),
                     marker='+', color='b', markerSize=12)
    plt.xticks(np.arange(len(inter_exp_biases)), xticks)
    plt.xlim([-0.5, len(inter_exp_biases)-0.5])
    f_bias.savefig(saving_folder_all + '/all_together_perf.png', dpi=DPI,
                   bbox_inches='tight')


def reduce_xticks(specs):
    specs = specs.replace('pass_reward', 'Rew')
    specs = specs.replace('pass_action', 'Act')
    specs = specs.replace('num_exps', 'N')
    specs = specs.replace('bl_dur', 'bl')
    specs = specs.replace('network', 'net')
    specs = specs.replace('nlstm', 'n_un')
    specs = specs.replace('twin_net', 'twin')
    specs = specs.replace('cont_rnn', 'rnn')
    return specs


def organize_biases(biases_after_seqs):
    biases = np.empty((len(biases_after_seqs),
                       num_trials_back-1, 4))
    biases_t_2 = np.empty((len(biases_after_seqs),
                           num_trials_back-1, 4))
    for ind_exp in range(len(biases_after_seqs)):
        mat_biases = biases_after_seqs[ind_exp]
        counter = 0
        for ind_perf in range(2):
            for ind_tr in range(2):
                index = np.logical_and(mat_biases[:, 1] == ind_perf,
                                       mat_biases[:, 2] == ind_tr)
                biases[ind_exp, :, counter] = mat_biases[index, 0]
                biases_t_2[ind_exp, :, counter] =\
                    mat_biases[index, 4]
                counter += 1
    return biases, biases_t_2


def sort_results(mat, expl_params):
    if len(expl_params) == 1:
        keys = list(expl_params.keys())
        mat = sorted(mat, key=lambda i: i[keys[0]])
    return mat


def get_main_results(file, bias_acr_training, biases_after_seqs,
                     biases_after_transEv, num_samples_mat, per,
                     perf_last_stage):
    choice, _, performance, evidence, _ =\
        load_behavioral_data(file)
    perf_last_stage.append(np.mean(performance[20000:].flatten()))
    # plot performance
    bias_mat = bias_across_training(choice, evidence,
                                    performance, per=per,
                                    conv_window=2)
    bias_acr_training.append(bias_mat)
    #
    mat_biases, mat_num_samples =\
        bias_after_altRep_seqs(file=file, num_tr=500000)
    biases_after_seqs.append(mat_biases)
    num_samples_mat.append(mat_num_samples)
    #
    mat_biases = bias_after_transEv_change(file=file,
                                           num_tr=500000)
    biases_after_transEv.append(mat_biases)
    return perf_last_stage, bias_acr_training, biases_after_seqs,\
        biases_after_transEv, num_samples_mat


def plot_main_results(file, bias_acr_training, biases_after_seqs,
                      biases_after_transEv, num_samples_mat, leg_flag, per):
    mat_conv = np.arange(1, num_trials_back)
    choice, correct_side, performance, evidence, _ =\
        load_behavioral_data(file)
    # plot performance
    num_tr = 10000000
    start_point = 0
    plt.subplot(3, 2, 1)
    plot_learning(performance[start_point:start_point+num_tr],
                  evidence[start_point:start_point+num_tr],
                  correct_side[start_point:start_point+num_tr],
                  w_conv=1000, legend=leg_flag)
    plt.subplot(3, 2, 2)
    bias_mat = bias_acr_training[-1]
    plot_bias_across_training(bias_mat,
                              tot_num_trials=choice.shape[0],
                              folder='',
                              fig=False, legend=leg_flag,
                              per=per, conv_window=2)
    #
    mat_biases = biases_after_seqs[-1]
    mat_num_samples = num_samples_mat[-1]
    plot_bias_after_altRep_seqs(mat_biases, mat_conv,
                                mat_num_samples,
                                folder='',
                                panels=[3, 2, 3],
                                legend=leg_flag)
    #
    mat_biases = biases_after_transEv[-1]
    plot_bias_after_transEv_change(mat_biases,
                                   folder='',
                                   panels=[3, 2, 5],
                                   legend=leg_flag)


def set_yaxis():
    maximo = -np.inf
    minimo = np.inf
    for ind_pl in range(3, 4):
        plt.subplot(3, 2, ind_pl)
        ax = plt.gca()
        lims = ax.get_ylim()
        maximo = max(maximo, lims[1])
        minimo = min(minimo, lims[0])
    for ind_pl in range(2, 7):
        plt.subplot(3, 2, ind_pl)
        ax = plt.gca()
        lims = ax.set_ylim([minimo, maximo])


if __name__ == '__main__':
    #    plt.close('all')
    #    per = 50000
    #    conv_window = 2
    #    mat_conv = np.arange(1, num_trials_back)
    #    folder = '/home/linux/all_results/supervised_RDM_' +\
    #        't_100_200_200_200_100_TH_0.2_0.8' +\
    #        '_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20_' +\
    #        'd_2KKK_ne_24_nu_32' +\
    #        '_ev_0.5_408140/'
    #    file = folder + '/bhvr_data_all.npz'
    #    data_flag = ptf.put_files_together(folder,
    #                                       min_num_trials=0)
    #    choice, correct_side, performance, evidence, _ =\
    #        load_behavioral_data(file)
    #    # plot performance
    #    bias_mat = bias_across_training(choice, evidence,
    #                                    performance, per=per,
    #                                    conv_window=conv_window)
    #    plot_bias_across_training(bias_mat,
    #                              tot_num_trials=choice.shape[0],
    #                              folder='',
    #                              fig=True, legend=True,
    #                              per=per, conv_window=conv_window)
    #    mat_biases, mat_num_samples =\
    #        bias_after_altRep_seqs(file=file, num_tr=per)
    #    plot_bias_after_altRep_seqs(mat_biases, mat_conv,
    #                                mat_num_samples,
    #                                folder='',
    #                                panels=None,
    #                                legend=True)
    #    #
    #    mat_biases = bias_after_transEv_change(file=file,
    #                                           num_tr=per)
    #    plot_bias_after_transEv_change(mat_biases,
    #                                   folder='',
    #                                   panels=None,
    #                                   legend=True)
    #    asdasd
    if len(sys.argv) > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = home + '/all_results/'
    batch_analysis(main_folder=main_folder, neural_analysis_flag=False,
                   behavior_analysis_flag=True)

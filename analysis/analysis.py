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
n_exps_fig_2_ccn = 50  # 217
acr_tr_per = 100000
acr_train_step = 400
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
colores = np.concatenate((azul.reshape((1, 3)), rojo.reshape((1, 3)),
                          verde.reshape((1, 3)), morado.reshape((1, 3)),
                          naranja.reshape((1, 3))), axis=0)

############################################
# AUXLIARY FUNCTIONS
############################################


def get_times(num, per, step):
    return np.linspace(0, num-per, (num-per)//step+1, dtype=int)


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
    **without taking into account the current trial**.
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


def compute_bias_perf_transHist(ch, ev, trans, perf, p_hist, conv_window,
                                figs=False, new_fig=False):
    values = np.unique(trans)
    biases = np.empty((2, 2))
    if figs:
        if new_fig:
            ut.get_fig(display_mode, font=8)
        labels = ['after error alt', 'after error rep',
                  'after correct alt', 'after correct rep']
        counter = 0
    for ind_perf in range(2):
        for ind_tr in [0, values.shape[0]-1]:
            ind_tr_ = int(ind_tr/(values.shape[0]-1))
            mask = np.logical_and.reduce((trans == values[ind_tr],
                                          perf == ind_perf,
                                          p_hist == conv_window))
            mask = np.concatenate((np.array([False]), mask[:-1]))
            if ind_perf == 0 and ind_tr == 0 and False:
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
                if figs:
                    if ind_tr_ == 0:
                        color = rojo
                    elif ind_tr_ == 1:
                        color = azul
                    x = np.linspace(np.min(ev),
                                    np.max(ev), 50)
                    label = labels[counter] + ' b: ' + str(round(popt[1], 3))
                    alpha = 1-0.5*(1-ind_perf)
                    plot_psycho_curve(x, popt, label, color, alpha)
                    counter += 1
            else:
                popt = [0, 0]
            biases[ind_perf, ind_tr_] = popt[1]
    return biases


def plot_psycho_curve(x, popt, label='', color=azul, alpha=1):
    # get the y values for the fitting
    y = probit_lapse_rates(x, popt[0], popt[1],
                           popt[2], popt[3])
    plt.plot(x, y, color=color,  label=label, lw=0.5,
             alpha=alpha)
    plt.legend(loc="lower right")
    plot_dashed_lines(-np.max(x), np.max(x))


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


def remove_top_right_axis():
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
############################################
# NEURAL ANALYSIS
############################################


def get_simulation_vars(file='/home/linux/network_data_492999.npz', fig=False,
                        n_envs=12, env=0, num_steps=100, obs_size=4,
                        num_units=128, num_act=3, num_steps_fig=200, start=0,
                        save_folder=''):
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
        rows = 4
        cols = 1
        lw = 1
        gris = (.6, .6, .6)
        trials_temp = trials[start:start+num_steps_fig]
        tr_time = np.where(trials_temp == 1)[0] + 0.5
        f = ut.get_fig(display_mode, font=16)
        # FIGURE
        # states
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
        gt_temp = np.argmax(gt, axis=0)
        plt.subplot(rows, cols, 3)
        for ind_tr in range(len(tr_time)):
            plt.plot(np.ones((2,))*tr_time[ind_tr], [0, 2], '--',
                     color=gris, lw=lw)
        plt.plot(actions[start:start+num_steps_fig], '-+', lw=lw,
                 color=colores[2, :])
        plt.plot(gt_temp[start:start+num_steps_fig], '--+', lw=lw,
                 color=colores[4, :])
        plt.xlim([-0.5, num_steps_fig-0.5])
        plt.ylabel('Action (gt)')
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
            ut.color_axis(ax, color=rojo)
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
    choice = data['choice']
    stimulus = data['stimulus']
    correct_side = data['correct_side']
    performance = (choice == correct_side)
    evidence = stimulus[:, 1] - stimulus[:, 2]
    return choice, correct_side, performance, evidence


def plot_learning(performance, evidence, stim_position, w_conv=200,
                  legend=False):
    """
    plots RNN and ideal observer performances.
    The function assumes that a figure has been created
    before it is called.
    """
    lw = 0.1
    num_trials = performance.shape[0]
    # remove all previous plots
    # ideal observer choice
    io_choice = (evidence < 0) + 1
    io_performance = io_choice == stim_position
    # save the mean performances
    RNN_perf = np.mean(performance[2000:].flatten())
    io_perf = np.mean(io_performance.flatten())

    # plot smoothed performance
    performance_smoothed = np.convolve(performance,
                                       np.ones((w_conv,))/w_conv,
                                       mode='valid')
    performance_smoothed = performance_smoothed[0::w_conv]
    plt.plot(np.linspace(0, num_trials, performance_smoothed.shape[0]),
             performance_smoothed, color=(0.39, 0.39, 0.39), lw=lw,
             label='RNN perf. (' + str(round(RNN_perf, 3)) + ')', alpha=0.5)
    # plot ideal observer performance
    plt.plot([0, num_trials], np.ones((2,))*io_perf, color=(1, 0.8, 0.5),
             lw=0.5,
             label='Ideal Obs. perf. (' + str(round(io_perf, 3)) + ')')
    # plot 0.25, 0.5 and 0.75 performance lines
    plot_fractions([0, performance.shape[0]])
    plt.title('performance')
    plt.xlabel('trials')
    if legend:
        plt.legend()


def bias_across_training(choice, evidence, performance,
                         per=100000, step=None, conv_window=3):
    """
    compute bias across training
    """
    if step is None:
        step = per
    steps = get_times(choice.shape[0], per, step)
    transitions = get_transition_mat(choice, conv_window=conv_window)
    perf_hist = np.convolve(performance, np.ones((conv_window,)),
                            mode='full')[0:-conv_window+1]
    perf_hist = np.concatenate((np.array([0]), perf_hist[:-1]))
    bias_mat = np.empty((len(steps), 2, 2))
    perf_mat = np.empty((len(steps)))
    periods_mat = np.empty((len(steps)))
    for ind, ind_per in enumerate(steps):
        ev = evidence[ind_per:ind_per+per]
        perf = performance[ind_per:ind_per+per]
        ch = choice[ind_per:ind_per+per]
        trans = transitions[ind_per:ind_per+per]
        p_hist = perf_hist[ind_per:ind_per+per]
        periods_mat[ind] = ind_per + per/2
        perf_mat[ind] = np.mean(perf)
        biases = compute_bias_perf_transHist(ch, ev, trans, perf,
                                             p_hist, conv_window)
        for ind_perf in range(2):
            for ind_tr in range(2):
                bias_mat[ind, ind_perf, ind_tr] = biases[ind_perf, ind_tr]
    return bias_mat, perf_mat, periods_mat


def plot_bias_across_training(bias_mat, tot_num_trials, folder='',
                              fig=True, legend=False, per=100000, step=None,
                              conv_window=3):
    """
    plot the results obtained by bias_across_training
    """
    if step is None:
        step = per
    time_stps = np.linspace(0, tot_num_trials-per,
                            (tot_num_trials-per)//step+1, dtype=int)
    lbl_perf = ['error', 'correct']
    lbl_trans = ['alts', 'reps']
    if fig:
        f = ut.get_fig(display_mode)
    for ind_perf in range(2):
        for ind_tr in range(2):
            if ind_tr == 0:
                color = rojo
            elif ind_tr == 1:
                color = azul
            bias = bias_mat[:, ind_perf, ind_tr]
            plt.plot(time_stps, bias, '+-', color=color, lw=1,
                     label=str(conv_window) + ' ' + lbl_trans[ind_tr] +
                     ' after ' + lbl_perf[ind_perf], alpha=0.5*(1+ind_perf))
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
    _, _, performance, evidence = load_behavioral_data(file)
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
    choice, _, performance, evidence = load_behavioral_data(file)
    # BIAS CONDITIONED ON TRANSITION HISTORY (NUMBER OF REPETITIONS)
    start_point = performance.shape[0]-num_tr
    ev = evidence[start_point:start_point+num_tr]
    perf = performance[start_point:start_point+num_tr]
    ch = choice[start_point:start_point+num_tr]
    mat_biases = []
    mat_conv = np.arange(1, num_trials_back)
    mat_num_samples = np.zeros((num_trials_back-1, 2, 2, 2))
    for conv_window in mat_conv:
        # get number of repetitions during the last conv_window trials
        # (not including the current trial)
        if conv_window > 1:
            trans = get_transition_mat(ch, conv_window=conv_window)
        else:
            repeat = get_repetitions(ch)
            trans = np.concatenate((np.array([0]), repeat[:-1]))
        # perf_hist is use to check that all previous last trials where correct
        if conv_window > 1:
            perf_hist = np.convolve(perf, np.ones((conv_window,)),
                                    mode='full')[0:-conv_window+1]
            perf_hist = np.concatenate((np.array([0]), perf_hist[:-1]))
        else:
            perf_hist = np.concatenate((np.array([0]), perf[:-1]))
        next_perf = np.concatenate((np.array([0]), perf[:-1]))
        values = np.unique(trans)
        for ind_perf in range(2):
            for ind_tr in [0, values.shape[0]-1]:
                for ind_curr_tr in range(2):
                    # mask finds all times in which the current trial is
                    # correct/error and the trial history (num. of repetitions)
                    # is values[ind_tr] we then need to shift these times
                    # to get the bias in the trial following them
                    mask = np.logical_and.reduce((trans == values[ind_tr],
                                                  perf == ind_perf,
                                                  perf_hist == conv_window,
                                                  repeat == ind_curr_tr))
                    mask = np.concatenate((np.array([False]), mask[:-1]))
                    if conv_window == 2 and ind_curr_tr == 0 and\
                       ind_perf == 1 and False:
                        index = np.where(mask != 0)[0]
                        num = 50
                        start = index[0] - 10
                        ut.get_fig(display_mode)
                        for ind in range(num):
                            plt.plot([ind, ind], [-2, 2], '--',
                                     color=(.6, .6, .6))
                        repeat = get_repetitions(ch)
                        plt.plot(repeat[start:start+num]-1, '-+', lw=1,
                                 label='repeat')
                        plt.plot(trans[start:start+num], '-+', lw=1,
                                 label='trans')
                        plt.plot(perf_hist[start:start+num], '-+', lw=1,
                                 label='perf_hist')
                        plt.plot(perf[start:start+num]-3, '-+', lw=1,
                                 label='performance')
                        plt.plot(mask[start:start+num]-3, '-+', lw=1,
                                 label='mask')
                        plt.legend()

                    mat_num_samples[conv_window-1,
                                    int(ind_tr/(values.shape[0]-1)),
                                    ind_perf, ind_curr_tr] += np.sum(mask)
                    if np.sum(mask) > 100:
                        popt, _ = bias_calculation(ch.copy(), ev.copy(),
                                                   mask.copy())
                    else:
                        popt = [0, 0]
                    # here I want to compute the bias at t+2 later
                    # when the trial 1 step later was correct
                    mask = np.logical_and.reduce((trans == values[ind_tr],
                                                  perf == ind_perf,
                                                  perf_hist == conv_window,
                                                  next_perf == 1,
                                                  repeat == ind_curr_tr))

                    mask = np.concatenate((np.array([False, False]),
                                           mask[:-2]))
                    if np.sum(mask) > 100:
                        popt_next, _ = bias_calculation(ch.copy(), ev.copy(),
                                                        mask.copy())
                    else:
                        popt_next = [0, 0]
                    mat_biases.append([popt[1], ind_perf,
                                       ind_tr/(values.shape[0]-1), ind_curr_tr,
                                       conv_window, popt_next[1]])
    mat_biases = np.array(mat_biases)
    return mat_biases, mat_num_samples


def plot_bias_after_altRep_seqs(mat_biases, mat_conv, mat_num_samples,
                                folder='', panels=None, legend=False):
    lbl_curr_tr = ['alt', 'rep']
    lbl_perf = ['error', 'correct']
    lbl_tr = ['alt', 'rep']
    if panels is None:
        f = ut.get_fig(display_mode, font=4)
    counter = 4
    for ind_c_tr in range(2):
        for ind_perf in range(2):
            counter -= 1
            if panels is None:
                plt.subplot(2, 2, counter+1)
            else:
                plt.subplot(panels[0], panels[1], panels[2]+counter)
            ax = plt.gca()
            ax.set_title('bias after ' + lbl_perf[ind_perf] +
                         ' + ' + lbl_curr_tr[ind_c_tr])
            for ind_tr in range(2):
                if ind_tr == 0:
                    color = rojo
                elif ind_tr == 1:
                    color = azul
                # plot number of samples per number of rep/alt transition
                ax2 = ax.twinx()
                ax2.plot(mat_conv, mat_num_samples[:, ind_tr,
                                                   ind_perf, ind_c_tr],
                         color=color, alpha=0.2)
                ax2.set_ylabel('total num. of samples')

                index = np.logical_and.reduce((mat_biases[:, 1] == ind_perf,
                                               mat_biases[:, 2] == ind_tr,
                                               mat_biases[:, 3] == ind_c_tr))
                ax.plot(mat_conv, mat_biases[index, 0], '+-', color=color,
                        lw=1,
                        label=lbl_tr[ind_tr] + ' + ' + lbl_perf[ind_perf])
                ax.plot(mat_conv, mat_biases[index, 5], '--', color=color,
                        lw=1,
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


def single_exp_analysis(file, exp, per, step, bias_acr_training=[],
                        biases_after_seqs=[], num_samples_mat=[],
                        performances=[], leg_flag=True,
                        fig=False, plot=True):
    if fig:
        ut.get_fig(display_mode, font=4)

    data_flag = ptf.put_files_together(exp, min_num_trials=per)
    if data_flag:
        performances, bias_acr_training, biases_after_seqs,\
            num_samples_mat = get_main_results(file, bias_acr_training,
                                               biases_after_seqs,
                                               num_samples_mat,
                                               per, step, performances)
        if plot:
            plot_main_results(file, bias_acr_training,
                              biases_after_seqs,
                              num_samples_mat, leg_flag, per, step)
    return bias_acr_training, biases_after_seqs,\
        num_samples_mat, performances, data_flag


def batch_analysis(main_folder, neural_analysis_flag=False,
                   behavior_analysis_flag=True):
    per = acr_tr_per
    step = acr_train_step
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
                                     rep_prob=params['rep_prob'],
                                     num_env=params['num_env'])
        folder = os.path.basename(os.path.normpath(folder + '/'))
        # only late experiments indicate the parameter alpha explicitly in the
        # name of the folder
        if 'alpha' not in expl_keys:
            undscr_ind = folder.rfind('_a_')
            folder_name = folder[:undscr_ind]
        else:
            undscr_ind = folder.rfind('_')
            folder_name = folder[:undscr_ind]
        n_envs_ind = folder_name.find('_ne_')
        folder_name =\
            folder_name.replace(folder_name[n_envs_ind:n_envs_ind+6], '*')
        files = glob.glob(main_folder + folder_name + '*')

        if len(files) > 0:
            f = ut.get_fig(display_mode)
            biases_after_seqs = []
            bias_acr_training = []
            num_samples_mat = []
            performances = []
            files_used = []
            for ind_f in range(len(files)):
                leg_flag = ind_f == 0
                file = files[ind_f] + '/bhvr_data_all.npz'
                print(ind_f)
                print(files[ind_f])
                bias_acr_training, biases_after_seqs,\
                    num_samples_mat, performances, data_flag =\
                    single_exp_analysis(file, files[ind_f], per, step,
                                        bias_acr_training, biases_after_seqs,
                                        num_samples_mat,
                                        performances, leg_flag)
                if data_flag:
                    files_used.append(files[ind_f])
            set_yaxis()
            # get biases
            biases, biases_t_2, non_cond_biases =\
                organize_biases(biases_after_seqs, bias_acr_training)
            p_exp['num_exps'] = biases.shape[0]
            results = {'biases_after_seqs': biases_after_seqs,
                       'bias_across_training': bias_acr_training,
                       'num_samples_mat': num_samples_mat,
                       'non_cond_biases': non_cond_biases,
                       'biases': biases,
                       'biases_t_2': biases_t_2,
                       'performances': performances,
                       'exps': files_used,
                       'p_exp': p_exp,
                       'per': per,
                       'step': step}
            np.savez(saving_folder_all + '/' + folder_name +
                     '_results.npz', **results)
            f.savefig(saving_folder_all + '/' + folder_name +
                      '_bhvr_fig.png', dpi=DPI, bbox_inches='tight')

            p_exp['biases'] = non_cond_biases
            p_exp['perfs'] = performances[-1]
            inter_exp_biases.append(p_exp)
    plot_biases_all_experiments(inter_exp_biases, expl_params,
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
        specs = json.dumps(p_exp)
        specs = reduce_xticks(specs)
        xticks.append(specs)
        for ind_perf in range(2):
            for ind_tr in range(2):
                if ind_tr == 0:
                    color = rojo
                elif ind_tr == 1:
                    color = azul
                mean = mat_means[ind_perf, ind_tr]
                std = mat_std[ind_perf, ind_tr]
                plt.plot(np.random.normal(loc=ind_exp,
                                          scale=0.01*len(inter_exp_biases),
                                          size=(all_biases.shape[0],)),
                         all_biases[:, ind_perf, ind_tr], '.', color=color,
                         markerSize=5, alpha=0.5*(1+ind_perf))
                plt.errorbar(ind_exp, mean, std/np.sqrt(num_exps),
                             marker='+', color=color, markerSize=10,
                             alpha=0.5*(1+ind_perf))
    plt.xticks(np.arange(len(inter_exp_biases)), xticks)
    plt.xlim([-0.5, len(inter_exp_biases)-0.5])
    f_bias.savefig(saving_folder_all + '/all_together_bias.pdf', dpi=DPI,
                   bbox_inches='tight')
    f_bias.savefig(saving_folder_all + '/all_together_bias.svg', dpi=DPI,
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
                 color=azul, markerSize=3, alpha=1.)
        plt.errorbar(ind_exp, mat_means, mat_std/np.sqrt(num_exps),
                     marker='+', color=azul, markerSize=12)
    plt.xticks(np.arange(len(inter_exp_biases)), xticks)
    plt.xlim([-0.5, len(inter_exp_biases)-0.5])
    f_bias.savefig(saving_folder_all + '/all_together_perf.svg', dpi=DPI,
                   bbox_inches='tight')
    f_bias.savefig(saving_folder_all + '/all_together_perf.pdf', dpi=DPI,
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


def organize_biases(biases_after_seqs, bias_acr_training):
    non_cond_biases = np.empty((len(biases_after_seqs), 2, 2))
    biases = np.empty((len(biases_after_seqs),
                       num_trials_back-1, 2, 2, 2))
    biases_t_2 = np.empty((len(biases_after_seqs),
                           num_trials_back-1, 2, 2, 2))
    for ind_exp in range(len(biases_after_seqs)):
        mat_b_non_cond = bias_acr_training[ind_exp]
        mat_b = biases_after_seqs[ind_exp]
        for ind_perf in range(2):
            for ind_tr in range(2):
                non_cond_biases[ind_exp, ind_perf, ind_tr] =\
                    mat_b_non_cond[-1, ind_perf, ind_tr]
                for ind_c_tr in range(2):
                    index = np.logical_and.reduce((mat_b[:, 1] == ind_perf,
                                                   mat_b[:, 2] == ind_tr,
                                                   mat_b[:, 3] == ind_c_tr))
                    biases[ind_exp, :, ind_perf, ind_tr, ind_c_tr] =\
                        mat_b[index, 0]
                    biases_t_2[ind_exp, :, ind_perf, ind_tr, ind_c_tr] =\
                        mat_b[index, 5]

    return biases, biases_t_2, non_cond_biases


def sort_results(mat, expl_params):
    if len(expl_params) == 1:
        keys = list(expl_params.keys())
        mat = sorted(mat, key=lambda i: i[keys[0]])
    return mat


def get_main_results(file, bias_acr_training, biases_after_seqs,
                     num_samples_mat, per, step,
                     perf_acr_training):
    choice, _, performance, evidence =\
        load_behavioral_data(file)
    # plot performance
    bias_mat, perf_mat, periods = bias_across_training(choice, evidence,
                                                       performance, per=per,
                                                       step=step,
                                                       conv_window=2)

    bias_acr_training.append(bias_mat)
    perf_acr_training.append(perf_mat)
    #
    mat_biases, mat_num_samples =\
        bias_after_altRep_seqs(file=file, num_tr=500000)
    biases_after_seqs.append(mat_biases)
    num_samples_mat.append(mat_num_samples)
    #
    return perf_acr_training, bias_acr_training, biases_after_seqs,\
        num_samples_mat


def plot_main_results(file, bias_acr_training, biases_after_seqs,
                      num_samples_mat, leg_flag, per, step):
    mat_conv = np.arange(1, num_trials_back)
    choice, correct_side, performance, evidence =\
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
                              per=per, step=step, conv_window=2)
    #
    mat_biases = biases_after_seqs[-1]
    mat_num_samples = num_samples_mat[-1]
    plot_bias_after_altRep_seqs(mat_biases, mat_conv,
                                mat_num_samples,
                                folder='',
                                panels=[3, 2, 3],
                                legend=leg_flag)


def set_yaxis():
    maximo = -np.inf
    minimo = np.inf
    for ind_pl in range(2, 7):
        plt.subplot(3, 2, ind_pl)
        ax = plt.gca()
        lims = ax.get_ylim()
        maximo = max(maximo, lims[1])
        minimo = min(minimo, lims[0])
    for ind_pl in range(2, 7):
        plt.subplot(3, 2, ind_pl)
        ax = plt.gca()
        lims = ax.set_ylim([minimo, maximo])


############################################
# FIGURES
############################################


def process_exp(bias_exps, perfs_exp, after_error_alt_all, after_error_rep_all,
                after_correct_alt_all, after_correct_rep_all, times, perfs_all,
                per, step, ax_main_panel, plt_b_acr_time, rows, cols, lw,
                alpha, labels, axis_lbs, leg_flag, max_train_duration,
                marker='.'):
    after_error_alt, after_error_rep, after_correct_alt,\
        after_correct_rep, after_error_alt_all, after_error_rep_all,\
        after_correct_alt_all, after_correct_rep_all, times, perfs_all =\
        accumulate_data(bias_exps, perfs_exp, after_error_alt_all,
                        after_error_rep_all, after_correct_alt_all,
                        after_correct_rep_all, times, perfs_all, per, step)
    plt.sca(ax_main_panel)
    plot_biases_acr_tr_allExps(after_error_alt, after_error_rep,
                               after_correct_alt, after_correct_rep,
                               labels, axis_lbs, max_train_duration,
                               leg_flag, alpha, marker=marker)
    # PLOT BIASES ACROSS TRAINING
    if plt_b_acr_time:
        times_aux = times[-after_error_rep.shape[0]:]
        plt.subplot(rows, cols, 2)
        plt.plot(times_aux, after_correct_rep, color=azul, alpha=alpha, lw=lw)
        plt.plot(times_aux, after_correct_alt, color=rojo, alpha=alpha, lw=lw)
        plt.subplot(rows, cols, 3)
        plt.plot(times_aux, after_error_rep, color=azul, alpha=alpha, lw=lw)
        plt.plot(times_aux, after_error_alt, color=rojo, alpha=alpha, lw=lw)
    return after_error_alt_all, after_error_rep_all,\
        after_correct_alt_all, after_correct_rep_all, times, perfs_all


def accumulate_data(bias_exps, perfs_exp, after_error_alt_all,
                    after_error_rep_all, after_correct_alt_all,
                    after_correct_rep_all, times, perfs_all, per, step):
    after_error_alt = bias_exps[:, 0, 0]
    after_error_rep = bias_exps[:, 0, 1]
    after_correct_alt = bias_exps[:, 1, 0]
    after_correct_rep = bias_exps[:, 1, 1]
    after_error_alt_all = np.concatenate((after_error_alt_all,
                                          after_error_alt))
    after_error_rep_all = np.concatenate((after_error_rep_all,
                                          after_error_rep))
    after_correct_alt_all = np.concatenate((after_correct_alt_all,
                                            after_correct_alt))
    after_correct_rep_all = np.concatenate((after_correct_rep_all,
                                            after_correct_rep))
    times = np.concatenate((times,
                            np.arange(after_error_alt.shape[0])*step+per/2))
    perfs_all = np.concatenate((perfs_all, perfs_exp))

    return after_error_alt, after_error_rep, after_correct_alt,\
        after_correct_rep, after_error_alt_all, after_error_rep_all,\
        after_correct_alt_all, after_correct_rep_all, times, perfs_all


def plot_hist_proj(after_error_alt_all, after_error_rep_all,
                   after_correct_alt_all, after_correct_rep_all,
                   xs, lw, b, f):
    ax = plt.gca()
    points = ax.get_position().get_points()
    width_height = np.diff(points, axis=0)[0]
    ax2 = f.add_axes([points[0][0], points[1][1],
                      width_height[0], width_height[1]/2])
    hist = np.histogram(after_error_alt_all, bins=xs)[0]
    plt.step(xs[:-1]+b/2, hist, color=rojo, lw=lw,
             label='After error alt')

    hist = np.histogram(after_error_rep_all, bins=xs)[0]
    plt.step(xs[:-1]+b/2, hist, color=azul, lw=lw,
             label='After error rep')
    ax2.axis('off')
    ax2 = f.add_axes([points[1][0], points[0][1],
                      width_height[0]/2, width_height[1]])
    hist = np.histogram(after_correct_alt_all, bins=xs)[0]
    plt.step(hist, xs[:-1]+b/2, color=rojo, lw=lw,
             label='After correct alt')

    hist = np.histogram(after_correct_rep_all, bins=xs)[0]
    plt.step(hist, xs[:-1]+b/2, color=azul, lw=lw,
             label='After correct rep')
    ax2.axis('off')


def plot_psychocurve_examples(ax1, ax2):
    # example reset after error
    conv_window = 2
    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
    folder = main_folder + 'supervised_RDM_t_100_200_200_200_100_' +\
        'TH_0.2_0.8_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_' +\
        'g_0.8_b_20_ne_24_nu_16_ev_0.5_a_0.1_865154/'
    file = folder + 'bhvr_data_all.npz'
    ch, _, perf, ev =\
        load_behavioral_data(file)
    ev = ev[-acr_tr_per:]
    perf = perf[-acr_tr_per:]
    ch = ch[-acr_tr_per:]
    trans = get_transition_mat(ch, conv_window=conv_window)
    p_hist = np.convolve(perf, np.ones((conv_window,)),
                         mode='full')[0:-conv_window+1]
    p_hist = np.concatenate((np.array([0]), p_hist[:-1]))
    plt.sca(ax1)
    compute_bias_perf_transHist(ch, ev, trans, perf, p_hist, conv_window,
                                figs=True, new_fig=False)
    ax1.set_xlim([-1, 1])
    ax1.get_legend().remove()
    ax1.set_xlabel('Repeating evidence')
    ax1.set_ylabel('Response probability')
    remove_top_right_axis()
    # example no reset after error
    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
    folder = main_folder + 'supervised_RDM_t_100_200_200_200_100_' +\
        'TH_0.2_0.8_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_' +\
        'g_0.8_b_20_ne_24_nu_16_ev_0.5_a_0.1_891427/'
    file = folder + 'bhvr_data_all.npz'
    ch, _, perf, ev =\
        load_behavioral_data(file)
    ev = ev[-acr_tr_per:]
    perf = perf[-acr_tr_per:]
    ch = ch[-acr_tr_per:]
    trans = get_transition_mat(ch, conv_window=conv_window)
    p_hist = np.convolve(perf, np.ones((conv_window,)),
                         mode='full')[0:-conv_window+1]
    p_hist = np.concatenate((np.array([0]), p_hist[:-1]))
    plt.sca(ax2)
    compute_bias_perf_transHist(ch, ev, trans, perf, p_hist, conv_window,
                                figs=True, new_fig=False)
    ax2.set_xlim([-1, 1])
    ax2.get_legend().remove()
    ax2.set_ylabel('Response probability')
    remove_top_right_axis()


def plot_biases_acr_tr_allExps(after_error_alt, after_error_rep,
                               after_correct_alt, after_correct_rep,
                               labels, axis_lbs, max_tr_dur, colors=None,
                               leg_flag=True, alpha=1.,
                               marker='.'):
    """

    """
    if colors is None:
        colors = colores
    pair1 = [after_error_rep, after_correct_rep]
    pair2 = [after_error_alt, after_correct_alt]
    if leg_flag:
        plt.plot(pair1[0], pair1[1], color=colors[0, :],
                 lw=0.1, label=labels[0])
        plt.plot(pair2[0], pair2[1], color=colors[1, :],
                 lw=0.1, label=labels[1])
        plt.xlabel(axis_lbs[0])
        plt.ylabel(axis_lbs[1])
        plt.plot([-6, 6], [0, 0], '--k', lw=0.2)
        plt.plot([0, 0], [-8, 8], '--k', lw=0.2)
        plt.plot([-6, 6], [8, -8], '--k', lw=0.2)
        plt.legend()
    else:
        plt.plot(pair1[0], pair1[1], color=colors[0, :],
                 lw=0.1, alpha=alpha)
        plt.plot(pair2[0], pair2[1], color=colors[1, :],
                 lw=0.1, alpha=alpha)
    if marker == 'x' or marker == '+':
        plt.plot(pair1[0][-1], pair1[1][-1], marker=marker, color='k',
                 alpha=pair1[0].shape[0]/max_tr_dur, markersize=3)
        plt.plot(pair2[0][-1], pair2[1][-1], marker=marker, color='k',
                 alpha=pair1[0].shape[0]/max_tr_dur, markersize=3)
    else:
        plt.plot(pair1[0][-1], pair1[1][-1], marker=marker,
                 color=colors[0, :], alpha=pair1[0].shape[0]/max_tr_dur,
                 markersize=3)
        plt.plot(pair2[0][-1], pair2[1][-1], marker=marker,
                 color=colors[1, :], alpha=pair1[0].shape[0]/max_tr_dur,
                 markersize=3)


def plot_mean_bias(pair1, pair2, xs, b, colores, invert=False, label='',
                   lw=1.5, alpha=1.):
    factor = 0.75
    medians_pair1 = []
    medians_pair2 = []
    for ind_bin in range(xs.shape[0]-1):
        medians_pair1 = bin_bias(medians_pair1, pair1,
                                 xs[ind_bin:ind_bin+2], b)
        medians_pair2 = bin_bias(medians_pair2, pair2,
                                 xs[ind_bin:ind_bin+2], b)
    medians_pair1 = np.array(medians_pair1)
    medians_pair2 = np.array(medians_pair2)
    if invert:
        plt.plot(medians_pair1[:, 1], medians_pair1[:, 0], label=label,
                 color=colores[0, :]*factor, lw=lw, alpha=alpha)
        plt.plot(medians_pair2[:, 1], medians_pair2[:, 0],
                 color=colores[1, :]*factor, lw=lw, alpha=alpha)
    else:
        plt.plot(medians_pair1[:, 0], medians_pair1[:, 1],
                 color=colores[0, :]*factor, lw=lw, alpha=alpha, label=label)
        plt.plot(medians_pair2[:, 0], medians_pair2[:, 1],
                 color=colores[1, :]*factor, lw=lw, alpha=alpha)


def bin_bias(medians, pair, binning, b):
    indx = np.logical_and(pair[1] > binning[0],
                          pair[1] <= binning[1])
    if np.sum(indx) > 0:
        medians.append([np.median(pair[0][indx]),
                        binning[0]+b/2])
#        print(binning)
#        print(binning[0]+b/2)
#        print(np.sum(indx))
#        print(np.mean(pair[0][indx]))
#        print('--------------------')
    return medians


def fig_2_ccn(file_all_exps, folder, pl_axis=[[-9, 9], [-12, 12]], b=0.8):
    f1 = ut.get_fig(font=8)
    margin_plt = 0.03
    margin = pl_axis[1][1]+b/2
    alpha = .5
    rows = 3
    cols = 3
    lw = 0.5
    # Panels, labels and other stuff for panel d
    loc_main_panel = [0.5, 0.2, 0.3, 0.3]
    ax_main_panel = f1.add_axes(loc_main_panel)
    loc_ex_panel = [0.2, 0.175, 0.15, 0.15]
    ax_ex_panel1 = f1.add_axes(loc_ex_panel)
    loc_ex_panel = [0.2, 0.425, 0.15, 0.15]
    ax_ex_panel2 = f1.add_axes(loc_ex_panel)
    xs = np.linspace(-margin, margin, int(2*margin/b+1))
    labels = ['Repeating context',
              'Alternating context']
    axis_lbs = ['After error bias', 'After correct bias']
    # PLOT PERFORMANCES
    plt.subplot(rows, cols, 1)
    num_tr = 10000000
    start_point = 0
    files = glob.glob(folder + '/supervised*')
    for ind_f in range(n_exps_fig_2_ccn):
        leg_flag = ind_f == 0
        file = files[ind_f] + '/bhvr_data_all.npz'
        if os.path.exists(file):
            choice, correct_side, performance, evidence =\
                load_behavioral_data(file)
            plot_learning(performance[start_point:start_point+num_tr],
                          evidence[start_point:start_point+num_tr],
                          correct_side[start_point:start_point+num_tr],
                          w_conv=1000, legend=leg_flag)
    ax = plt.gca()
    ax.get_legend().remove()
    plt.legend(['RNNs', 'Perfect integrator'])
    ax.set_title('')
    remove_top_right_axis()
    ax.set_ylabel('Performance')
    ax.set_xlabel('Trials   ')
    ax.set_xticks([0, 500000, 1000000])
    ax.set_xticklabels(['0', '0.5', '1 (M)'])
    points = ax.get_position().get_points()
    plt.text(points[0][0]-margin_plt, points[1][1]+margin_plt, 'a',
             transform=plt.gcf().transFigure)

    # PLOT BIAS ACROSS TRAINING
    data = np.load(file_all_exps)
    files = data['exps']
    bias_acr_tr = data['bias_across_training']
    perfs = data['performances']
    per = data['per']
    step = data['step']
    max_train_duration = max([x.shape[0] for x in bias_acr_tr])
    after_error_alt_all = np.empty((0,))
    after_error_rep_all = np.empty((0,))
    after_correct_alt_all = np.empty((0,))
    after_correct_rep_all = np.empty((0,))
    times = np.empty((0,))
    perfs_all = np.empty((0,))
    for ind_exp in range(len(bias_acr_tr)):
        if (ind_exp != 137 and ind_exp != 19) and perfs[ind_exp][-1] > 0.6:
            bias_exp = bias_acr_tr[ind_exp]
            perfs_exp = perfs[ind_exp]
            after_error_alt_all, after_error_rep_all, after_correct_alt_all,\
                after_correct_rep_all, times, perfs_all =\
                process_exp(bias_exp, perfs_exp,
                            after_error_alt_all, after_error_rep_all,
                            after_correct_alt_all, after_correct_rep_all,
                            times, perfs_all, per, step, ax_main_panel,
                            ind_exp < n_exps_fig_2_ccn,
                            rows, cols, lw, alpha, labels, axis_lbs,
                            ind_exp == 0, max_train_duration, marker='.')
    plt.xlim(pl_axis[0])
    plt.ylim(pl_axis[1])
    pair1 = [after_error_rep_all, after_correct_rep_all]
    pair2 = [after_error_alt_all, after_correct_alt_all]
    plot_mean_bias(pair1, pair2, xs, b, colores)
    remove_top_right_axis()
    ind_exp = 19
    bias_exp = bias_acr_tr[ind_exp]
    perfs_exp = perfs[ind_exp]
    after_error_alt_all, after_error_rep_all, after_correct_alt_all,\
        after_correct_rep_all, times, perfs_all =\
        process_exp(bias_exp, perfs_exp,
                    after_error_alt_all, after_error_rep_all,
                    after_correct_alt_all, after_correct_rep_all,
                    times, perfs_all, per, step, ax_main_panel,
                    ind_exp < n_exps_fig_2_ccn,
                    rows, cols, lw, alpha, labels, axis_lbs,
                    ind_exp == 0, max_train_duration, marker='x')
    ind_exp = 137
    bias_exp = bias_acr_tr[ind_exp]
    perfs_exp = perfs[ind_exp]
    after_error_alt_all, after_error_rep_all, after_correct_alt_all,\
        after_correct_rep_all, times, perfs_all =\
        process_exp(bias_exp, perfs_exp,
                    after_error_alt_all, after_error_rep_all,
                    after_correct_alt_all, after_correct_rep_all,
                    times, perfs_all, per, step, ax_main_panel,
                    ind_exp < n_exps_fig_2_ccn,
                    rows, cols, lw, alpha, labels, axis_lbs,
                    ind_exp == 0, max_train_duration, marker='+')

    # PLOT PROJECTIONS
    plot_hist_proj(after_error_alt_all, after_error_rep_all,
                   after_correct_alt_all, after_correct_rep_all,
                   xs, lw, b, f1)
    # PLOT MEANS AND TUNE PANEL B
    plt.subplot(rows, cols, 2)
    xs_across_training = np.unique(times)
    b_across_training = step
    pair1 = [after_correct_rep_all, times]
    pair2 = [after_correct_alt_all, times]
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colores, invert=True)
    ax = plt.gca()
    ylim = ax.get_ylim()
    remove_top_right_axis()
    plt.ylabel('After correct bias')
    plt.xlabel('Trials')
    ax.set_xticks([0, 500000, 1000000])
    ax.set_xticklabels(['0', '0.5', '1 (M)'])
    points = ax.get_position().get_points()
    plt.text(points[0][0]-margin_plt, points[1][1]+margin_plt, 'b',
             transform=plt.gcf().transFigure)

    # PLOT MEANS AND TUNE PANEL c
    plt.subplot(rows, cols, 3)
    pair1 = [after_error_rep_all, times]
    pair2 = [after_error_alt_all, times]
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colores, invert=True)
    ax = plt.gca()
    ylim = ax.set_ylim(ylim)
    remove_top_right_axis()
    plt.ylabel('After error bias')
    plt.xlabel('Trials')
    ax.set_xticks([0, 500000, 1000000])
    ax.set_xticklabels(['0', '0.5', '1 (M)'])
    points = ax.get_position().get_points()
    plt.text(points[0][0]-margin_plt, points[1][1]+margin_plt, 'c',
             transform=plt.gcf().transFigure)

    # PLOT PSYCHOCURVE EXAMPLES
    plot_psychocurve_examples(ax_ex_panel1, ax_ex_panel2)
    ax = plt.gca()
    points = ax.get_position().get_points()
    plt.text(points[0][0]-2*margin_plt, points[1][1]+margin_plt, 'd',
             transform=plt.gcf().transFigure)

    f1.savefig('/home/molano/priors/figures_CCN/Fig2.svg', dpi=DPI,
               bbox_inches='tight')
    f1.savefig('/home/molano/priors/figures_CCN/Fig2.pdf', dpi=DPI,
               bbox_inches='tight')


def plot_2d_fig_biases(file, plot_all=False, fig=None, leg_flg=False,
                       pl_axis=[[-9, 9], [-12, 12]], b=1, colors=None):
    if fig is None:
        f = ut.get_fig(font=8)
    if colors is None:
        colors = colores
    margin = pl_axis[1][1]+b/2
    alpha = .5
    lw = 0.5
    # Panels, labels and other stuff for panel d
    xs = np.linspace(-margin, margin, int(2*margin/b+1))
    labels = ['Repeating context',
              'Alternating context']
    axis_lbs = ['After error bias', 'After correct bias']
    # PLOT BIAS ACROSS TRAINING
    data = np.load(file)
    bias_acr_tr = data['bias_across_training']
    p_exp = data['p_exp']
    perfs = data['performances']
    per = data['per']
    step = data['step']
    specs = json.dumps(p_exp.tolist())
    specs = reduce_xticks(specs)
    max_train_duration = max([x.shape[0] for x in bias_acr_tr])
    after_error_alt_all = np.empty((0,))
    after_error_rep_all = np.empty((0,))
    after_correct_alt_all = np.empty((0,))
    after_correct_rep_all = np.empty((0,))
    times = np.empty((0,))
    perfs_all = np.empty((0,))
    loc_main_panel = [0.3, 0.2, 0.4, 0.4]
    if fig is None:
        f.add_axes(loc_main_panel)
    for ind_exp in range(len(bias_acr_tr)):
        if perfs[ind_exp][-1] > 0.6:
            bias_exps = bias_acr_tr[ind_exp]
            perfs_exp = perfs[ind_exp]
            after_error_alt, after_error_rep, after_correct_alt,\
                after_correct_rep, after_error_alt_all, after_error_rep_all,\
                after_correct_alt_all, after_correct_rep_all,\
                _, perfs_all =\
                accumulate_data(bias_exps, perfs_exp, after_error_alt_all,
                                after_error_rep_all, after_correct_alt_all,
                                after_correct_rep_all, times, perfs_all,
                                per, step)
            if plot_all:
                plot_biases_acr_tr_allExps(after_error_alt, after_error_rep,
                                           after_correct_alt,
                                           after_correct_rep,
                                           labels, axis_lbs,
                                           max_train_duration, colors=colors,
                                           (ind_exp == 0 and leg_flg), alpha,
                                           marker='.')
    plt.xlim(pl_axis[0])
    plt.ylim(pl_axis[1])
    pair1 = [after_error_rep_all, after_correct_rep_all]
    pair2 = [after_error_alt_all, after_correct_alt_all]
    plot_mean_bias(pair1, pair2, xs, b, colores, label=specs, lw=0.5)
    remove_top_right_axis()
    plt.xlabel(axis_lbs[0])
    plt.ylabel(axis_lbs[1])
    plt.plot([-pl_axis[0][1], pl_axis[0][1]], [0, 0], '--k', lw=0.2)
    plt.plot([0, 0], [-pl_axis[1][1], pl_axis[1][1]], '--k', lw=0.2)
    plt.plot([-pl_axis[0][1], pl_axis[0][1]], [pl_axis[1][1], -pl_axis[1][1]],
             '--k', lw=0.2)
    if plot_all:
        plot_hist_proj(after_error_alt_all, after_error_rep_all,
                       after_correct_alt_all, after_correct_rep_all,
                       xs, lw, b, f)


def plot_2d_fig_diff_params(main_folder, order=None, save_folder='',
                            pl_axis=[[-9, 9], [-12, 12]], name=''):
    folder = main_folder + 'all_results/'
    files = glob.glob(folder + '*results.npz')
    f = ut.get_fig(font=8)
    if order is None:
        order = np.arange(len(files))
    for ind_f in range(len(files)):
        print(files[order[ind_f]])
        plt.subplot(3, len(files)*2, len(files)*2+2*ind_f+1)
        plot_2d_fig_biases(files[order[ind_f]], plot_all=True, fig=f,
                           leg_flg=False, pl_axis=pl_axis)
    if save_folder != '':
        f.savefig(save_folder + '/' + name + '_2d_plots.svg', dpi=DPI,
                  bbox_inches='tight')
        f.savefig(save_folder + '/' + name + '_2d_plots.pdf', dpi=DPI,
                  bbox_inches='tight')


def plot_biases_diff_parameterss(file, means, perfs_means,
                                 plot_all=False, f=None, leg_flg=False,
                                 b=1, ind=0):
    data = np.load(file)
    bias_acr_tr = data['bias_across_training']
    perfs = data['performances']
    means_temp = np.empty((len(bias_acr_tr), 4))
    perfs_temp = []
    for ind_exp in range(len(bias_acr_tr)):
        perfs_temp.append(perfs[ind_exp][-1])
        exp = bias_acr_tr[ind_exp]
        after_error_alt = exp[:, 0, 0][-1]
        after_error_rep = exp[:, 0, 1][-1]
        after_correct_alt = exp[:, 1, 0][-1]
        after_correct_rep = exp[:, 1, 1][-1]
        markersize = 2
        plt.plot(ind, after_correct_alt, color=rojo, marker='.',
                 markersize=markersize)
        plt.plot(ind, after_error_alt, color=rojo, marker='.',
                 markersize=markersize, alpha=0.4)
        plt.plot(ind, after_correct_rep, color=azul, marker='.',
                 markersize=markersize)
        plt.plot(ind, after_error_rep, color=azul, marker='.',
                 markersize=markersize, alpha=0.4)
        means_temp[ind_exp, :] = [after_correct_alt, after_error_alt,
                                  after_correct_rep, after_error_rep]
    means.append(np.mean(means_temp, axis=0))
    perfs_means.append([np.mean(perfs_temp),
                        np.std(perfs_temp)/np.sqrt(len(perfs_temp))])
    return means, perfs_means


def plot_fig_diff_params(main_folder, save_folder=''):
    f = ut.get_fig(font=8)
    list_exps = ['repeating_probability', 'num_neurons', 'block_size',
                 'pass_reward_action']
    order_all = [[0, 2, 4, 1, 3], [0, 2, 3, 1], [7, 3, 6, 2, 1, 5, 0, 4],
                 [1, 0, 3, 2]]
    xticks = [['0.5-0.5', '0.6-0.4', '0.7-0.3', '0.8-0.2', '0.9-0.1'],
              ['8 unts', '16 unts', '32 unts', '64 unts'],
              ['10', '40', '100', '200', '300', '400', '1000', '10000'],
              ['-', 'Act', 'Rew', 'Rew+Act']]
    xlabel = ['Repeating probability', 'Number of units', 'Block size',
              'Extra information']
    panels = 'abcd'
    for ind_exp in range(len(list_exps)):
        plt.subplot(4, 2, 2*(ind_exp+1) + (ind_exp+1) % 2)
        folder = main_folder + list_exps[ind_exp] + '/all_results/'
        files = glob.glob(folder + '*results.npz')
        print(files)
        print('-----------------')
        order = order_all[ind_exp]
        means = []
        perfs_means = []
        for ind_f in range(len(files)):
            print(files[order[ind_f]])
            print('xxxxxxxxxxxxxxx')
            means, perfs_means =\
                plot_biases_diff_parameterss(files[order[ind_f]], means,
                                             perfs_means,
                                             plot_all=True, f=f,
                                             leg_flg=False,
                                             ind=ind_f)
        means = np.array(means)
        perfs_means = np.array(perfs_means)
        markersize = 4
        plt.plot(means[:, 0], '-+', color=rojo, markersize=markersize)
        plt.plot(means[:, 1], '-+', color=rojo, markersize=markersize,
                 alpha=0.4)
        plt.plot(means[:, 2], '-+', color=azul, markersize=markersize)
        plt.plot(means[:, 3], '-+', color=azul, markersize=markersize,
                 alpha=0.4)
        plt.xlabel(xlabel[ind_exp])
        plt.ylabel('Bias')
        ax = plt.gca()
        ax.set_xticks(np.arange(len(files)))
        ax.set_xticklabels(xticks[ind_exp])
        remove_top_right_axis()
        # add axis for performances
        ax = plt.gca()
        points = ax.get_position().get_points()
        width_height = np.diff(points, axis=0)[0]
        ax2 = f.add_axes([points[0][0]+0.0, points[1][1]+0.01,
                          width_height[0], width_height[1]])
        remove_top_right_axis()
        plt.errorbar(np.arange(perfs_means.shape[0]),
                     perfs_means[:, 0], perfs_means[:, 1],
                     color='k', lw=1)
        ax2.set_ylim([min(perfs_means[:, 0])-0.05,
                      max(perfs_means[:, 0])+0.05])
        points = ax2.get_position().get_points()
        margin_plt = 0.05
        plt.text(points[0][0]-margin_plt, points[1][1]+margin_plt/2,
                 panels[ind_exp], transform=plt.gcf().transFigure)
        plt.ylabel('Performance')
        ax2.set_xticks([])
    if save_folder != '':
        f.savefig(save_folder + '/Fig3_diff_params.svg', dpi=DPI,
                  bbox_inches='tight')
        f.savefig(save_folder + '/Fig3_diff_params.pdf', dpi=DPI,
                  bbox_inches='tight')


def plot_2d_fig_biases_VS_perf(file, pl_axis=[[-12, 12], [0.5, 1]], b=1,
                               dur_th=2000, aft_err_th=2, perf_th=.7,
                               eq_fact=1, plot_all=True, f_p=None, alpha=1.):
    f_b = ut.get_fig(font=8)
    if f_p is None:
        f_p = ut.get_fig(font=8)
        loc_main_panel = [0.3, 0.2, 0.4, 0.4]
        f_p.add_axes(loc_main_panel)
    rows = 2
    cols = 2
    lw = 0.2
    margin = pl_axis[0][1]+b/2
    # Panels, labels and other stuff for panel d
    xs = np.linspace(-margin, margin, int(2*margin/b+1))
#    labels = ['Repeating context',
#              'Alternating context']
    # axis_lbs = ['After error bias', 'After correct bias']
    # PLOT BIAS ACROSS TRAINING
    data = np.load(file)
    bias_acr_tr = data['bias_across_training']
    p_exp = data['p_exp']
    perfs = data['performances']
    per = data['per']
    step = data['step']
    last_time_point = dur_th*step+per/2
    specs = json.dumps(p_exp.tolist())
    specs = reduce_xticks(specs)
    max_train_duration = max([x.shape[0] for x in bias_acr_tr])
#    durations = np.unique([x.shape[0] for x in bias_acr_tr])
#    print(durations)
#    print(durations.shape)
#    print(np.histogram([x.shape[0] for x in bias_acr_tr], bins=durations))
#    asdsd
    b_across_training = acr_tr_per
    xs_across_training = np.linspace(-acr_tr_per/2,
                                     acr_tr_per*(max_train_duration+1/2),
                                     max_train_duration+2)
    after_error_alt_all = np.empty((0,))
    after_error_rep_all = np.empty((0,))
    after_correct_alt_all = np.empty((0,))
    after_correct_rep_all = np.empty((0,))
    times = np.empty((0,))
    perfs_all = np.empty((0,))

    counter = 0
    for ind_exp in range(len(bias_acr_tr)):
        bias_exps = bias_acr_tr[ind_exp]
        perfs_exp = perfs[ind_exp]
        if perfs_exp.shape[0] > dur_th and perfs_exp[-1] > perf_th and\
           eq_fact*np.abs(bias_exps[-1, 0, 0]) > eq_fact*aft_err_th and\
           eq_fact*np.abs(bias_exps[-1, 0, 1]) > eq_fact*aft_err_th:
            counter += 1
            after_error_alt, after_error_rep, after_correct_alt,\
                after_correct_rep, after_error_alt_all, after_error_rep_all,\
                after_correct_alt_all, after_correct_rep_all,\
                times, perfs_all =\
                accumulate_data(bias_exps, perfs_exp, after_error_alt_all,
                                after_error_rep_all, after_correct_alt_all,
                                after_correct_rep_all, times, perfs_all,
                                per, step)
            if plot_all:
                plt.figure(f_b.number)
                plt.subplot(1, 2, 1)
                plt.plot(after_correct_rep, perfs_exp, color=azul)
                plt.plot(after_correct_alt, perfs_exp, color=rojo)
                plt.xlabel('after correct')
                plt.ylabel('performance')
                plt.subplot(1, 2, 2)
                plt.plot(after_error_rep, perfs_exp, color=azul)
                plt.plot(after_error_alt, perfs_exp, color=rojo)
                plt.xlabel('after error')
                plt.ylabel('performance')

                plt.figure(f_p.number)
                times_aux = times[-perfs[ind_exp].shape[0]:]
                plt.subplot(rows, cols, 1)
                plt.plot(times_aux[:dur_th], perfs[ind_exp][:dur_th], '-',
                         color=(.7, .7, .7), lw=lw, alpha=0.5*alpha)
                plt.subplot(rows, cols, 2)
                plt.plot(times_aux[:dur_th], perfs[ind_exp][:dur_th], '-',
                         color=(.7, .7, .7), lw=lw, alpha=0.5*alpha)
                plt.subplot(rows, cols, 3)
                plt.plot(times_aux[:dur_th], after_correct_rep[:dur_th],
                         color=azul, lw=lw, alpha=0.5*alpha)
                plt.plot(times_aux[:dur_th], after_correct_alt[:dur_th],
                         color=rojo, lw=lw, alpha=0.5*alpha)
                plt.subplot(rows, cols, 4)
                plt.plot(times_aux[:dur_th], after_error_rep[:dur_th],
                         color=azul, lw=lw, alpha=0.5*alpha)
                plt.plot(times_aux[:dur_th], after_error_alt[:dur_th],
                         color=rojo, lw=lw, alpha=0.5*alpha)

    plt.figure(f_b.number)
    plt.subplot(1, 2, 1)
    pair1 = [perfs_all, after_correct_rep_all]
    pair2 = [perfs_all, after_correct_alt_all]
    plot_mean_bias(pair1, pair2, xs, b, colores, invert=True, alpha=alpha)
    remove_top_right_axis()

    plt.subplot(1, 2, 2)
    pair1 = [perfs_all, after_error_rep_all]
    pair2 = [perfs_all, after_error_alt_all]
    plot_mean_bias(pair1, pair2, xs, b, colores, invert=True, alpha=alpha)
    remove_top_right_axis()

    plt.figure(f_p.number)
    pair1 = [perfs_all, times]
    pair2 = [perfs_all, times]
    xs_across_training = np.unique(times)
    xs_across_training = xs_across_training[xs_across_training <=
                                            last_time_point]
    b_across_training = step
    last_event = np.floor(last_time_point/per) + 1
    ylim = [.7, .82]
    colors = np.zeros((2, 3))
    plt.subplot(rows, cols, 1)
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colors, invert=True, alpha=alpha,
                   label='median perf. (N=: ' + str(counter) + ')')
    plt.legend()
    plt.plot([per/2, last_time_point], [0.709, 0.709], color=(1, .8, .5))
    plot_time_event(np.arange(1, last_event)*per)
    plt.ylim(ylim)
    remove_top_right_axis()
    plt.subplot(rows, cols, 2)
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colors, invert=True, alpha=alpha)
    plt.plot([per/2, last_time_point], [0.709, 0.709], color=(1, .8, .5))
    plot_time_event(np.arange(1, last_event)*per)
    plt.ylim(ylim)
    remove_top_right_axis()
    plt.subplot(rows, cols, 3)
    pair1 = [after_correct_rep_all, times]
    pair2 = [after_correct_alt_all, times]
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colores, invert=True, alpha=alpha)
    plt.plot([per/2, last_time_point], [0, 0], color=(.5, .5, .5))
    plot_time_event(np.arange(1, last_event)*per)
    remove_top_right_axis()
    plt.subplot(rows, cols, 4)
    pair1 = [after_error_rep_all, times]
    pair2 = [after_error_alt_all, times]
    plot_mean_bias(pair1, pair2, xs_across_training, b_across_training,
                   colores, invert=True, alpha=alpha)
    plt.plot([per/2, last_time_point], [0, 0], color=(.5, .5, .5))
    plot_time_event(np.arange(1, last_event)*per)
    remove_top_right_axis()
    return f_p


if __name__ == '__main__':
    plt.close('all')
    save_folder = '/home/molano/priors/figures_CCN/'

    # PLOT BIAS DIFFERENT PARAMETERS
#    main_folder = '/home/molano/priors/results/'
#    plot_fig_diff_params(main_folder, save_folder)
    # asasd
    # PLOT EXPERIMENT STRUCTURE
#    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
#    file = main_folder + 'supervised_RDM_t_100_200_200_200_100_' +\
#        'TH_0.2_0.8_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_' +\
#        'g_0.8_b_20_ne_24_nu_16_ev_0.5_a_0.1_891427/network_data_124999.npz'
#    get_simulation_vars(file=file, fig=True, n_envs=24, env=0, num_steps=20,
#                        obs_size=5, num_units=16, num_act=3,
#                        num_steps_fig=24, start=1002,
#                        save_folder=save_folder)
    # PLOT 2D FIG NUM. NEURONS
#    main_folder = '/home/molano/priors/results/num_neurons/'
#    plot_2d_fig_diff_params(main_folder, order=[0, 2, 3, 1],
#                            save_folder=save_folder,
#                            pl_axis=[[-9, 9], [-12, 12]], name='num_neurons')
#    asd
    # PLOT BIASES VERSUS PERFORMANCE
    b = 0.2
    pl_axis = [[-12, 12], [0.5, 1]]
    dur_th = 1500
    perf_th = .7
    plot_all = False
    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
    folder = main_folder + 'all_results/'
    file = folder + 'supervised_RDM_t_100_200_200_200_100_TH_0.2_0.8_200_' +\
        'PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20*_' +\
        'nu_16_ev_0.5_results.npz'
    fp = plot_2d_fig_biases_VS_perf(file, pl_axis=pl_axis, b=b,
                                    dur_th=dur_th, aft_err_th=2,
                                    perf_th=perf_th,
                                    eq_fact=1, plot_all=plot_all, f_p=None)
    fp = plot_2d_fig_biases_VS_perf(file, pl_axis=pl_axis, b=b,
                                    dur_th=dur_th, aft_err_th=2,
                                    perf_th=perf_th,
                                    eq_fact=-1, plot_all=plot_all,
                                    f_p=fp, alpha=0.3)
    fp.savefig(save_folder+'/perf_and_bias_' + str(dur_th) + '_' +
               str(perf_th) + '_' + str(plot_all) + '.svg', dpi=DPI,
               bbox_inches='tight')
    fp.savefig(save_folder+'/perf_and_bias_' + str(dur_th) + '_' +
               str(perf_th) + '_' + str(plot_all) + '.pdf', dpi=DPI,
               bbox_inches='tight')
    # PLOT 2D FIG 16-UNIT NETWORKS
#    asd
    b = 0.8
    pl_axis = [[-9, 9], [-12, 12]]
    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
    folder = main_folder + 'all_results/'
    file = folder + 'supervised_RDM_t_100_200_200_200_100_TH_0.2_0.8_200_' +\
        'PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20*_' +\
        'nu_16_ev_0.5_results.npz'
    fig_2_ccn(file, main_folder, pl_axis=pl_axis, b=b)
    plot_2d_fig_biases(file, plot_all=True, fig=None, leg_flg=True,
                       pl_axis=pl_axis, b=b)
#    asd
    if len(sys.argv) > 1:
        main_folder = sys.argv[1]
        files = glob.glob(main_folder + '/*')
    else:
        main_folder = home + '/priors/results/'
        files = glob.glob(home + '/priors/results/*')
    print(files)
    # index = 1 --> pass reward/action
    # index = 3 --> num_neurons
    # index = 4 --> rollout
    # index = 6 --> repeating probability
    # index = 7 --> 16-unit nets
    # index = 11 --> a2c pass reward/action
    list_exps = [3]  # [1, 3, 4, 6, 7, 8, 9, 10, 11]  # [3, 6, 7]
    for ind_f in range(len(files)):
        if ind_f in list_exps:
            plt.close('all')
            print(files[ind_f])
            batch_analysis(main_folder=files[ind_f]+'/',
                           neural_analysis_flag=False,
                           behavior_analysis_flag=True)

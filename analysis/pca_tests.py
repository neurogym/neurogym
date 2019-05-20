#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:14:24 2019

@author: molano
"""
from neurogym.analysis import analysis as an
from neurogym.ops import utils as ut
from neurogym.ops import put_together_files as ptf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
dt = 100
conv_window = 6
bs = 4
rw = 8
per = 500000


def plot_pca(comps, times_aux, color, ax, num_stps_bck):
    mat = []
    times_aux = times_aux[times_aux > num_stps_bck]
    assert times_aux.shape[0] > 0
    for ind_t in range(times_aux.shape[0]):
        mat.append([comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 0],
                    comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 1],
                    comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 2]])

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    mat = np.array(mat)
    print(mat.shape)
    mat_mean = np.mean(mat, axis=0)
    ax.plot(mat_mean[0, :],
            mat_mean[1, :],
            mat_mean[2, :],
            c=color[0], marker='.')
    ax.scatter(mat_mean[0, 0],
               mat_mean[1, 0],
               mat_mean[2, 0],
               c=color[1], marker='o', s=10)
    ax.scatter(mat_mean[0, -1],
               mat_mean[1, -1],
               mat_mean[2, -1],
               c=color[1], marker='x', s=10)


def find_peaks(trans, peaks_ind):
    # find peaks in transition vector
    bl_change = np.concatenate((np.array([-201]), peaks_ind))
    bl_change = np.where(np.diff(bl_change) > 200)[0]
    peaks = []
    for ind_bl in range(bl_change.shape[0]-1):
        pks_ind_temp = peaks_ind[bl_change[ind_bl]:bl_change[ind_bl+1]]
        pks_temp = trans[pks_ind_temp]
        max_ind = np.argmax(pks_temp)
        peaks.append(pks_ind_temp[max_ind])
    peaks = np.array(peaks)
    return peaks


def pca():
    index = np.linspace(-dt*bs, dt*rw, int(bs+rw), endpoint=False)
    main_folder = '/home/molano/priors/results/pass_reward_action/'
    folder = main_folder + 'supervised_RDM_t_100_200_200_200_100_' +\
        'TH_0.2_0.8_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20' +\
        '_ne_24_nu_32_ev_0.5_a_0.1_243839/'
    file = folder + '/bhvr_data_all.npz'
    data_flag = ptf.put_files_together(folder, min_num_trials=per)
    print(data_flag)
    if data_flag:
        choice, _, performance, evidence, _ =\
            an.load_behavioral_data(file)
        # plot performance
#        bias_mat = an.bias_across_training(choice, evidence,
#                                           performance, per=per,
#                                           conv_window=2)
#        an.plot_bias_across_training(bias_mat,
#                                     tot_num_trials=choice.shape[0],
#                                     folder='',
#                                     fig=False, legend=True,
#                                     per=per, conv_window=2)
        file = folder + '/network_data_224999.npz'
        states, rewards, actions, _, trials, _, _ =\
            an.get_simulation_vars(file=file, fig=False,
                                   n_envs=24, env=0, num_steps=20, obs_size=5,
                                   num_units=32, num_act=3, num_steps_fig=100)
        reset = np.where(np.sum(states, axis=0) < 0.37)[0]
        # select period between 'dones'
        period = 300000
        states = states[:, reset[1]-period:reset[1]-1]
        rewards = rewards[reset[1]-period:reset[1]-1]
        actions = actions[reset[1]-period:reset[1]-1]
        trials = trials[reset[1]-period:reset[1]-1]
#        an.transition_analysis(file=file,
#                               fig=True, n_envs=24, env=0, num_steps=20,
#                               obs_size=5, num_units=32, window=(-5, 10),
#                               part=[[0, 32]], p_lbl=['all'], folder='')
        times = np.where(trials == 1)[0]
        choice = actions[times-1]
        perf = rewards[times]

        # PCA analysis over all states
        num_comps = 3
        pca = PCA(n_components=num_comps)
        pca.fit(states.T)
        comps = pca.transform(states.T)
        trans = an.get_transition_mat(choice, conv_window=200)  # conv_window)
        peaks_ind = np.where(trans > np.percentile(trans, 75))[0]
        peaks_max = find_peaks(trans, peaks_ind)
        # plot components
        fig = ut.get_fig()
        ax1 = fig.gca(projection='3d')
        num_stps_back = 801
        plot_pca(comps, times[peaks_max], 'rm', ax1, num_stps_back)
        # minima
        peaks_ind = np.where(trans < np.percentile(trans, 25))[0]
        peaks_min = find_peaks(trans, peaks_ind)
        # plot components
        plot_pca(comps, times[peaks_min], 'bc', ax1, num_stps_back)

        # PCA analysis over states at the end of the trial
        num_comps = 3
        #        pca = PCA(n_components=num_comps)
        #        pca.fit(states_tr.T)
        #        comps = pca.transform(states_tr.T)
        comps = comps[times]
        # plot components
        fig = ut.get_fig()
        ax2 = fig.gca(projection='3d')
        num_stps_back = 205
        plot_pca(comps, peaks_max, 'rm', ax2, num_stps_back)
        # minima
        # plot components
        plot_pca(comps, peaks_min, 'bc', ax2, num_stps_back)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_zlim(ax1.get_zlim())

        # conditioned on switch in transition evidence
        num_stps_back = 20
        fig = ut.get_fig()
        ax3 = fig.gca(projection='3d')
        trans = an.get_transition_mat(choice, conv_window=conv_window)
        kernel_fsw = np.arange(conv_window+1) - conv_window/2
        full_switch = np.convolve(trans, kernel_fsw,
                                  mode='full')[0:-conv_window]
        mask = full_switch == np.min(full_switch)
        times_aux = np.where(mask > 0)[0]
        plot_pca(comps, times_aux, 'rm', ax3, num_stps_back)
        mask = full_switch == np.max(full_switch)
        times_aux = np.where(mask > 0)[0]
        plot_pca(comps, times_aux, 'bc', ax3, num_stps_back)
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_ylim(ax1.get_ylim())
        ax3.set_zlim(ax1.get_zlim())

        # plot trans. evidence, components and states
        start = 0  # 98000
        num_p = 100000  # 2000
        mask = full_switch == np.max(full_switch)
        ut.get_fig()
        plt.subplot(3, 1, 1)
        plt.plot(trans[start:start+num_p], label='trans')
        plt.plot(full_switch[start:start+num_p], label='full_switch')
        plt.plot(10*mask[start:start+num_p], label='mask')
        plt.legend()
        plt.subplot(3, 1, 2)
        plt.plot(comps[start:start+num_p, 0], label='comp 1')
        plt.plot(comps[start:start+num_p, 1], label='comp 2')
        plt.plot(comps[start:start+num_p, 2], label='comp 3')
        plt.subplot(3, 1, 3)
        plt.imshow(states[:, start:start+num_p], aspect='auto')
        plt.legend()

        asasdasd

        ut.get_fig()
        rows = 3
        cols = 3
        plt.plot(comps[:, 0], comps[:, 1], '.')
        print(pca.explained_variance_ratio_)
        mat_pcs = np.empty((num_comps, bs+rw, 2, 2))
        for ind_perf in range(2):
            for ind_tr in [0, values.shape[0]-1]:
                ind_tr_ = int(ind_tr/(values.shape[0]-1))
                mask = np.logical_and.reduce((trans == values[ind_tr],
                                              perf == ind_perf,
                                              p_hist == conv_window))
                mask = np.concatenate((np.array([False]), mask[:-1]))
                times_aux = times[mask]
                print(np.sum(mask))
                if ind_perf == 1 and ind_tr == values.shape[0]-1 and False:
                    ut.get_fig()
                    num_points = 100
                    plt.subplot(2, 1, 1)
                    plt.plot(perf[:num_points], '+-', label='rews')
                    plt.plot(p_hist[:num_points], '+-', label='perf-hist')
                    plt.plot(mask[:num_points], '+-', label='mask')
                    for ind in range(num_points):
                        plt.plot([ind, ind], [-3, 3], '--',
                                 color=(.7, .7, .7))
                    plt.legend()

                    plt.subplot(2, 1, 2)
                    plt.plot(choice[:num_points], '+-', label='choice')
                    plt.plot(trans[:num_points], '+-', label='trans')
                    plt.plot(mask[:num_points], '+-', label='mask')
                    for ind in range(num_points):
                        plt.plot([ind, ind], [-3, 3], '--',
                                 color=(.7, .7, .7))
                    plt.legend()
                    # sdf

                for comp in range(num_comps):
                    aux = np.empty((bs+rw, times_aux.shape[0]))
                    for ind_t in range(times_aux.shape[0]):
                        aux[:, ind_t] =\
                            comps[times_aux[ind_t]-bs:
                                  times_aux[ind_t]+rw, comp]
                    mat_pcs[comp, :, ind_perf,
                            ind_tr_] = np.mean(aux, axis=1)

        f1 = ut.get_fig()
        f2 = ut.get_fig()
        ax = f2.gca(projection='3d')
        for ind_perf in range(2):
            for ind_tr in range(2):
                plt.figure(f2.number)
                ax.scatter(mat_pcs[0, :, ind_perf, ind_tr],
                           mat_pcs[1, :, ind_perf, ind_tr],
                           mat_pcs[2, :, ind_perf, ind_tr],
                           color=(1-ind_tr, 0, ind_tr), lw=1.,
                           alpha=1.-0.75*(1-ind_perf))
                plt.figure(f1.number)
                for comp in range(num_comps):
                    plt.subplot(rows, cols, comp+1)
                    plt.plot(index, mat_pcs[comp, :, ind_perf, ind_tr],
                             color=(1-ind_tr, 0, ind_tr), lw=1.,
                             alpha=1.-0.75*(1-ind_perf))
                    # asdsad
        #            plt.plot(comps[times_aux[ind_t]-bs:times_aux[ind_t]+rw, 0],
        #                     comps[times_aux[ind_t]-bs:times_aux[ind_t]+rw, 1],
        #                     '.', color=(1-ind_tr/(values.shape[0]-1), 0,
        #                                 ind_tr/(values.shape[0]-1)),
        #                     alpha=1.-0.5*(1-ind_perf))
        #            plt.plot(comps[times_aux[ind_t], 0],
        #                     comps[times_aux[ind_t], 1],
        #                     '+', color='k')
    
    
if __name__ == '__main__':
    pca()

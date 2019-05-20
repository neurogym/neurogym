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
conv_window = 4
bs = 4
rw = 8
per = 500000


def plot_pca(comps, times_aux, color, ax, num_stps_bck):
    mat = []
    times_aux = times_aux[times_aux > num_stps_bck]
    for ind_t in range(times_aux.shape[0]):
        print(comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 0].shape)
        mat.append([comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 0],
                    comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 1],
                    comps[times_aux[ind_t]-num_stps_bck:times_aux[ind_t], 2]])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    mat = np.array(mat)
    print(mat.shape)
    mat_mean = np.mean(mat, axis=0)
    ax.plot(mat_mean[0, :],
            mat_mean[1, :],
            mat_mean[2, :],
            c=color, marker='.')
    ax.scatter(mat_mean[0, 0],
               mat_mean[1, 0],
               mat_mean[2, 0],
               c='g', marker='+')
    ax.scatter(mat_mean[0, -1],
               mat_mean[1, -1],
               mat_mean[2, -1],
               c='k', marker='+')


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
        bias_mat = an.bias_across_training(choice, evidence,
                                           performance, per=per,
                                           conv_window=2)
        an.plot_bias_across_training(bias_mat,
                                     tot_num_trials=choice.shape[0],
                                     folder='',
                                     fig=False, legend=True,
                                     per=per, conv_window=2)
        file = folder + '/network_data_224999.npz'
        states, rewards, actions, ev, trials, gt, pi =\
            an.get_simulation_vars(file=file, fig=True,
                                   n_envs=24, env=0, num_steps=20, obs_size=5,
                                   num_units=32, num_act=3, num_steps_fig=100)
#        an.transition_analysis(file=file,
#                               fig=True, n_envs=24, env=0, num_steps=20,
#                               obs_size=5, num_units=32, window=(-5, 10),
#                               part=[[0, 32]], p_lbl=['all'], folder='')
        times = np.where(trials == 1)[0]
        # trial_int = np.mean(np.diff(times))*dt
        choice = actions[times-1]
        # outcome = rewards[times]
        perf = rewards[times]

        trans = an.get_transition_mat(choice, conv_window=conv_window)

        values = np.unique(trans)
        p_hist = np.convolve(perf, np.ones((conv_window,)),
                             mode='full')[0:-conv_window+1]
        p_hist = np.concatenate((np.array([0]), p_hist[:-1]))

        num_comps = 3
        pca = PCA(n_components=num_comps)
        pca.fit(states.T)
        comps = pca.transform(states.T)
        fig = ut.get_fig()
        ax = fig.gca(projection='3d')
        num_p = 20000
        ax.scatter(comps[:num_p, 0], comps[:num_p, 1], comps[:num_p, 2], '.')
        asd
        kernel_fsw = np.arange(conv_window+1) - conv_window/2
        full_switch = np.convolve(trans, kernel_fsw,
                                  mode='full')[0:-conv_window]
        mask = full_switch == np.max(full_switch)
        times_aux = times[mask]
        fig = ut.get_fig()
        ax = fig.gca(projection='3d')
        num_stps_back = 10000
        plot_pca(comps, times_aux, 'r', ax, num_stps_back)
        mask = full_switch == np.min(full_switch)
        times_aux = times[mask]
        plot_pca(comps, times_aux, 'b', ax, num_stps_back)
        ut.get_fig()
        plt.plot(mask)
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
                            comps[times_aux[ind_t]-bs:times_aux[ind_t]+rw, comp]
                    mat_pcs[comp, :, ind_perf, ind_tr_] = np.mean(aux, axis=1)
    
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

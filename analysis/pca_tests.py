#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:14:24 2019

@author: molano
"""
from neurogym.analysis import analysis as an
from neurogym.ops import utils as ut
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.close('all')
dt = 100
conv_window = 4
bs = 4
rw = 8


def pca():
    index = np.linspace(-dt*bs, dt*rw, int(bs+rw), endpoint=False)
    main_folder = '/home/molano/priors/results/16_neurons_100_instances/'
    folder = main_folder + 'supervised_RDM_t_100_200_200_200_100_' +\
        'TH_0.2_0.8_200_PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_' +\
        'g_0.8_b_20_ne_24_nu_16_ev_0.5_a_0.1_848368/'

    file = folder + '/network_data_124999.npz'
    states, rewards, actions, ev, trials, gt, pi =\
        an.get_simulation_vars(file=file, fig=True,
                               n_envs=24, env=0, num_steps=20, obs_size=5,
                               num_units=16, num_act=3, num_steps_fig=1000)
    times = np.where(trials == 1)[0]
    # trial_int = np.mean(np.diff(times))*dt
    choice = actions[times-1]
    # outcome = rewards[times]
    perf = rewards[times]

    # num_steps = trials.shape[0]
    trans = an.get_transition_mat(choice, conv_window=conv_window)
    values = np.unique(trans)
    p_hist = np.convolve(perf, np.ones((conv_window,)),
                         mode='full')[0:-conv_window+1]
    p_hist = np.concatenate((np.array([0]), p_hist[:-1]))

    num_comps = 9
    rows = 3
    cols = 3
    pca = PCA(n_components=num_comps)
    pca.fit(states.T)
    comps = pca.transform(states.T)
    ut.get_fig()
    plt.plot(comps[:, 0], comps[:, 1], '.')
    asdsa
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
            if ind_perf == 1 and ind_tr == values.shape[0]-1:
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
                sdf

            for comp in range(9):
                aux = np.empty((bs+rw, times_aux.shape[0]))
                for ind_t in range(times_aux.shape[0]):
                    aux[:, ind_t] =\
                        comps[times_aux[ind_t]-bs:times_aux[ind_t]+rw, comp]
                mat_pcs[comp, :, ind_perf, ind_tr_] = np.mean(aux, axis=1)

    ut.get_fig()
    for ind_perf in range(2):
        for ind_tr in range(2):
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

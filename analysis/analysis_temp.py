#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 14:38:14 2019

@author: molano
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from os.path import expanduser
home = expanduser("~")
sys.path.append(home + '/neurogym')
sys.path.append(home + '/mm5514/')
from neurogym.ops import utils as ut
from neurogym.analysis import analysis as an
DPI = 400

plt.close('all')

folder = '/home/molano/priors/results/16_neurons_100_instances/all_results/'
data = np.load(folder +
               'supervised_RDM_t_100_200_200_200_100_TH_0.2_0.8_200_' +
               'PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20*_' +
               'nu_16_ev_0.5_results.npz')
files = data['exps']
biases = data['non_cond_biases']
f0 = ut.get_fig(font=5)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 0, 0], '.', color='r', markerSize=5, alpha=0.25)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 0, 1], '.', color='b', markerSize=5, alpha=0.25)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 1, 0], '.', color='r', markerSize=5, alpha=1.)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 1, 1], '.', color='b', markerSize=5, alpha=1.)
plt.xlim([0, 2])
f1 = ut.get_fig(font=5)
b = 1
margin = 8.5
xs = np.linspace(-margin, margin, int(2*margin/b+1))
hist = np.histogram(biases[:, 0, 0], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, 'r', lw=1, alpha=0.25, label='after error alt')

hist = np.histogram(biases[:, 0, 1], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, 'b', lw=1, alpha=0.25, label='after error rep')

hist = np.histogram(biases[:, 1, 0], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, 'r', lw=1, alpha=1., label='after correct alt')

hist = np.histogram(biases[:, 1, 1], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, 'b', lw=1, alpha=1., label='after correct rep')
plt.legend()
plt.xlabel('history bias')
plt.ylabel('count')
bias_acr_tr = data['bias_across_training']
max_train_duration = max([x.shape[0] for x in bias_acr_tr])
f2 = ut.get_fig(font=5)
f3 = ut.get_fig(font=5)
for ind_exp in range(len(bias_acr_tr)-1):
    exp = bias_acr_tr[ind_exp]
    after_error_alt = exp[:, 0, 0]

    after_error_rep = exp[:, 0, 1]

    after_correct_alt = exp[:, 1, 0]

    after_correct_rep = exp[:, 1, 1]
    after_corr_sum = (abs(after_correct_alt[-1]) + abs(after_correct_rep[-1]))
    after_err_sum = (abs(after_error_alt[-1]) + abs(after_error_rep[-1]))
    mean_ratio = after_err_sum / after_corr_sum

    if abs(mean_ratio) > 0.8 and after_correct_rep.shape[0] > 9 and\
       (after_corr_sum) > 1:
        print(ind_exp)
        print(mean_ratio)
        print(after_corr_sum)
        print(after_correct_rep.shape[0])
        print(files[ind_exp])
        print('-----------')
    plt.figure(f2.number)
    an.plot_biases_acrr_tr_all_exps(after_error_rep, after_correct_rep,
                                    after_error_alt, after_correct_alt,
                                    pl_axis=[[-8, 8], [-8, 8]],
                                    max_tr_dur=max_train_duration,
                                    leg_flag=ind_exp == 0)
    # zoom
    plt.figure(f3.number)
    an.plot_biases_acrr_tr_all_exps(after_error_rep, after_correct_rep,
                                    after_error_alt, after_correct_alt,
                                    pl_axis=[[-3, 3], [-3, 3]],
                                    max_tr_dur=max_train_duration,
                                    leg_flag=ind_exp == 0)
f2.savefig(folder + 'biases_evolution.png', dpi=DPI, bbox_inches='tight')
f3.savefig(folder + 'biases_evolution_zoom.png', dpi=DPI, bbox_inches='tight')

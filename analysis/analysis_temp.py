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
print(biases.shape)
plt.figure()
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 0, 0], '.', color='r', markerSize=5, alpha=0.5)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 0, 1], '.', color='b', markerSize=5, alpha=0.5)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 1, 0], '.', color='r', markerSize=5, alpha=1.)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 1, 1], '.', color='b', markerSize=5, alpha=1.)
plt.xlim([0, 2])
plt.figure()
b = 1
num_trans = 3
margin = 8
xs = np.linspace(-margin, margin, int(2*margin/b+1))
hist = np.histogram(biases[:, 0, 0], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='r', alpha=0.5)

hist = np.histogram(biases[:, 0, 1], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='b', alpha=0.5)

hist = np.histogram(biases[:, 1, 0], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='r', alpha=1.)

hist = np.histogram(biases[:, 1, 1], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='b', alpha=1.)


plt.figure()
bias_acr_tr = data['bias_across_training']
max_train_duration = max([x.shape[0] for x in bias_acr_tr])
bias_mean = []
for ind_exp in range(len(bias_acr_tr)-1):
    exp = bias_acr_tr[ind_exp]
    after_error_alt = exp[:, 0, 0]
    plt.plot(after_error_alt, color='r', alpha=0.5)

    after_error_rep = exp[:, 0, 1]
    plt.plot(after_error_rep, color='b', alpha=0.5)

    after_correct_alt = exp[:, 1, 0]
    plt.plot(after_correct_alt, color='r', alpha=1.)

    after_correct_rep = exp[:, 1, 1]
    plt.plot(after_correct_rep, color='b', alpha=1.)
    bias_mean.append([after_error_alt, after_error_rep,
                      after_correct_alt, after_correct_rep])

bias_mean = np.array(bias_mean)
f1 = ut.get_fig(font=5)
f2 = ut.get_fig(font=5)
for ind_exp in range(len(bias_acr_tr)-1):
    exp = bias_acr_tr[ind_exp]
    after_error_alt = exp[:, 0, 0]

    after_error_rep = exp[:, 0, 1]

    after_correct_alt = exp[:, 1, 0]

    after_correct_rep = exp[:, 1, 1]
    after_corr_sum = (abs(after_correct_alt[-1]) + abs(after_correct_rep[-1]))
    after_err_sum = (abs(after_error_alt[-1]) + abs(after_error_rep[-1]))
    mean_ratio = after_err_sum / after_corr_sum

    if abs(mean_ratio) < 0.05 and after_correct_rep.shape[0] > 9 and\
       (after_corr_sum) > 1:
        print(ind_exp)
        print(mean_ratio)
        print(after_corr_sum)
        print(after_correct_rep.shape[0])
        print(files[ind_exp])
        print('-----------')
    plt.figure(f1.number)
    an.plot_biases_acrr_tr_all_exps(after_error_rep, after_correct_rep,
                                    after_error_alt, after_correct_alt,
                                    pl_axis=[[-8, 8], [-8, 8]],
                                    max_tr_dur=max_train_duration)
    # zoom
    plt.figure(f2.number)
    an.plot_biases_acrr_tr_all_exps(after_error_rep, after_correct_rep,
                                    after_error_alt, after_correct_alt,
                                    pl_axis=[[-3, 3], [-3, 3]],
                                    max_tr_dur=max_train_duration)
f1.savefig(folder + 'biases_evolution.png', dpi=DPI, bbox_inches='tight')
f2.savefig(folder + 'biases_evolution_zoom.png', dpi=DPI, bbox_inches='tight')

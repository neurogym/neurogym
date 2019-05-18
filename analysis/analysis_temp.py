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
DPI = 400

plt.close('all')


def plot_biases(after_error_rep, after_correct_rep,
                after_error_alt, after_correct_alt,
                pl_axis=[[-6, 6], [-8, 8]],
                max_tr_dur=11):
    plt.subplot(2, 2, 1)
    colores = 'br'
    labels = ['after error rep VS after correct rep',
              'after error alt VS after correct alt']
    axis_lbs = ['after error bias', 'after correct bias']
    pair1 = [after_error_rep, after_correct_rep]
    pair2 = [after_error_alt, after_correct_alt]
    plot_biases_core(pair1, pair2, labels, axis_lbs, colores,
                     max_tr_dur=max_tr_dur)
    plt.xlim(pl_axis[0])
    plt.ylim(pl_axis[1])

    plt.subplot(2, 2, 2)
    colores = 'kg'
    labels = ['after error rep VS after correct alt',
              'after error alt VS after correct rep']
    axis_lbs = ['after error bias', 'after correct bias']
    pair1 = [after_error_rep, after_correct_alt]
    pair2 = [after_error_alt, after_correct_rep]
    plot_biases_core(pair1, pair2, labels, axis_lbs, colores,
                     max_tr_dur=max_tr_dur)
    plt.xlim(pl_axis[0])
    plt.ylim(pl_axis[1])

    plt.subplot(2, 2, 3)
    colores = 'kg'
    labels = ['after error rep VS after error alt',
              'after correc rep VS after correct alt']
    axis_lbs = ['after rep bias', 'after alt bias']
    pair1 = [after_error_rep, after_error_alt]
    pair2 = [after_correct_rep, after_correct_alt]
    plot_biases_core(pair1, pair2, labels, axis_lbs, colores,
                     max_tr_dur=max_tr_dur)
    plt.xlim(pl_axis[0])
    plt.ylim(pl_axis[1])


def plot_biases_core(pair1, pair2, labels, axis_lbs, colores, max_tr_dur):
    if ind_exp == 0:
        plt.plot(pair1[0], pair1[1], colores[0],
                 lw=0.1, label=labels[0])
        plt.plot(pair2[0], pair2[1], colores[1],
                 lw=0.1, label=labels[1])
        plt.xlabel(axis_lbs[0])
        plt.ylabel(axis_lbs[1])
        plt.plot([-6, 6], [0, 0], '--k', lw=0.2)
        plt.plot([0, 0], [-8, 8], '--k', lw=0.2)
        plt.legend()
    else:
        plt.plot(pair1[0], pair1[1], colores[0],
                 lw=0.1)
        plt.plot(pair2[0], pair2[1], colores[1],
                 lw=0.1)

    plt.plot(pair1[0][-1], pair1[1][-1], '.'+colores[0],
             alpha=pair1[0].shape[0]/max_tr_dur)
    plt.plot(pair2[0][-1], pair2[1][-1], '.'+colores[1],
             alpha=pair1[0].shape[0]/max_tr_dur)


folder = '/home/molano/priors/results/16_neurons_100_instances/'
data = np.load(folder +
               'supervised_RDM_t_100_200_200_200_100_TH_0.2_0.8_200_' +
               'PR_PA_cont_rnn_ec_0.05_lr_0.001_lrs_c_g_0.8_b_20_ne_24_' +
               'nu_16_ev_0.5_results.npz')
biases = data['biases']
plt.figure()
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 2, 0], '.', color='r', markerSize=5, alpha=0.5)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 2, 1], '.', color='b', markerSize=5, alpha=0.5)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 2, 2], '.', color='r', markerSize=5, alpha=1.)
plt.plot(np.random.normal(loc=1, scale=0.01, size=(biases.shape[0],)),
         biases[:, 2, 3], '.', color='b', markerSize=5, alpha=1.)
plt.xlim([0, 2])
plt.figure()
b = 1
num_trans = 3
margin = 8
xs = np.linspace(-margin, margin, 2*margin/b+1)
hist = np.histogram(biases[:, num_trans, 0], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='r', alpha=0.5)

hist = np.histogram(biases[:, num_trans, 1], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='b', alpha=0.5)

hist = np.histogram(biases[:, num_trans, 2], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='r', alpha=1.)

hist = np.histogram(biases[:, num_trans, 3], bins=xs)[0]
plt.plot(xs[:-1]+b/2, hist, color='b', alpha=1.)


plt.figure()
bias_acr_tr = data['bias_across_training']
max_train_duration = max([x.shape[0]/4 for x in bias_acr_tr])
bias_mean = []
for ind_exp in range(len(bias_acr_tr)):
    exp = bias_acr_tr[ind_exp]
    index = np.logical_and(exp[:, 1] == 0,
                           exp[:, 2] == 0)
    after_error_alt = exp[index, 0]
    plt.plot(after_error_alt, color='r', alpha=0.5)

    index = np.logical_and(exp[:, 1] == 0,
                           exp[:, 2] == 1)
    after_error_rep = exp[index, 0]
    plt.plot(after_error_rep, color='b', alpha=0.5)

    index = np.logical_and(exp[:, 1] == 1,
                           exp[:, 2] == 0)
    after_correct_alt = exp[index, 0]
    plt.plot(after_correct_alt, color='r', alpha=1.)

    index = np.logical_and(exp[:, 1] == 1,
                           exp[:, 2] == 1)
    after_correct_rep = exp[index, 0]
    plt.plot(after_correct_rep, color='b', alpha=1.)
    bias_mean.append([after_error_alt, after_error_rep,
                      after_correct_alt, after_correct_rep])

bias_mean = np.array(bias_mean)
f1 = ut.get_fig(font=5)
f2 = ut.get_fig(font=5)
for ind_exp in range(len(bias_acr_tr)):
    exp = bias_acr_tr[ind_exp]
    index = np.logical_and(exp[:, 1] == 0,
                           exp[:, 2] == 0)
    after_error_alt = exp[index, 0]

    index = np.logical_and(exp[:, 1] == 0,
                           exp[:, 2] == 1)
    after_error_rep = exp[index, 0]

    index = np.logical_and(exp[:, 1] == 1,
                           exp[:, 2] == 0)
    after_correct_alt = exp[index, 0]

    index = np.logical_and(exp[:, 1] == 1,
                           exp[:, 2] == 1)
    after_correct_rep = exp[index, 0]
    plt.figure(f1.number)
    plot_biases(after_error_rep, after_correct_rep,
                after_error_alt, after_correct_alt,
                pl_axis=[[-8, 8], [-8, 8]], max_tr_dur=max_train_duration)
    plt.figure(f2.number)
    plot_biases(after_error_rep, after_correct_rep,
                after_error_alt, after_correct_alt,
                pl_axis=[[-3, 3], [-3, 3]],  max_tr_dur=max_train_duration)
f1.savefig(folder + 'biases_evolution.png', dpi=DPI, bbox_inches='tight')
f2.savefig(folder + 'biases_evolution_zoom.png', dpi=DPI, bbox_inches='tight')

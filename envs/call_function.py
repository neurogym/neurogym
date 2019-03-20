#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:20:00 2019

@author: linux
"""
import os


def num2str(num):
    string = ''
    while num/1000 >= 1:
        string += 'K'
        num = num/1000
    string = str(int(num)) + string
    return string


if __name__ == '__main__':
    alg = 'a2c'
    env = 'RDM-v0'
    net = 'cont_rnn'
    nsteps = 100
    tot_num_stps = 1e10
    li = 1e2
    ent_coef = 0.1
    lr = 1e-3
    lr_sch = 'constant'
    gamma = 0.9
    num_env = 12
    trial_hist = True
    pass_reward = True
    if trial_hist:
        tr_h_flag = 'trHist'
    else:
        tr_h_flag = ''

    if pass_reward:
        pss_rw_flag = 'pssRew'
    else:
        pss_rw_flag = ''

    save_path = '../' + alg + '_' + env + '_' + tr_h_flag + '_' + pss_rw_flag +\
        '_' + net + '_ent_coef_' + str(ent_coef) + '_lr_' + str(lr) +\
        '_lrsch_' + lr_sch + '_g_' + str(gamma) +\
        '_batch_' + num2str(nsteps) + '_dur_' + num2str(tot_num_stps) +\
        '_n_env_' + str(num_env)
    save_path = save_path.replace('-v0', '')
    save_path = save_path.replace('constant', 'const')
    save_path = save_path.replace('linear', 'lin')
    # load_path = save_path + '/checkpoints/00020'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    command = 'python -m baselines.run --alg=' + alg + ' --env=' + env +\
        ' --network=' + net + ' --nsteps=' + str(nsteps) +\
        ' --num_timesteps=' + str(tot_num_stps) +\
        ' --log_interval=' + str(li) + ' --ent_coef=' + str(ent_coef) +\
        ' --lrschedule=' + lr_sch + ' --gamma=' + str(gamma) +\
        ' --num_env=' + str(num_env) + ' --trial_hist=' + str(trial_hist) +\
        ' --pass_reward=' + str(pass_reward) + ' --lr=' + str(lr) +\
        ' --save_path=' + save_path
    # + ' --load_path=' + load_path
    print(command)
    os.system(command)

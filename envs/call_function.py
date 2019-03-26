#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:20:00 2019

@author: linux
"""
import os
import utils as ut
import numpy as np

if __name__ == '__main__':
    alg = 'a2c'
    env = 'RDM-v0'
    net = 'cont_rnn'
    nsteps = 20
    tot_num_stps = 1e9
    li = 5e3
    ent_coef = 0.1
    lr = 1e-3
    lr_sch = 'constant'
    gamma = 0.9
    num_env = 12
    nlstm = 32

    if env == 'RDM-v0':
        timing = [100, 0, 0, 0, 100]
        timing_flag = '_t_' + ut.list_str(timing)
        timing_cmmd = ' --timing '
        for ind_t in range(len(timing)):
            timing_cmmd += str(timing[ind_t]) + ' '
    else:
        timing_flag = ''
        timing_cmmd = ''
    # trial history
    trial_hist = True
    if trial_hist:
        rep_prob = (.2, .8)
        bl_dur = 200
        tr_h_flag = '_TH_' + ut.list_str(rep_prob) + '_' + str(bl_dur)
        tr_h_cmmd = ' --trial_hist=True --bl_dur=' + str(bl_dur) +\
            ' --rep_prob '
        for ind_rp in range(len(rep_prob)):
            tr_h_cmmd += str(rep_prob[ind_rp]) + ' '
    else:
        tr_h_flag = ''
        tr_h_cmmd = ''
    # pass reward
    pass_reward = True
    if pass_reward:
        ps_rw_flag = '_PR'
        ps_rw_cmmd = ' --pass_reward=True'
    else:
        ps_rw_flag = ''
        ps_rw_cmmd = ''
    # pass action
    pass_action = True
    if pass_action:
        ps_a_flag = '_PA'
        ps_a_cmmd = ' --pass_action=True'
    else:
        ps_a_flag = ''
        ps_a_cmmd = ''

    load_path = ''
    if load_path == '':
        load_path_cmmd = ''
    else:
        load_path_cmmd = ' --load_path=' + load_path

    save_path = '../' + alg + '_' + env + timing_flag + \
        tr_h_flag + ps_rw_flag + ps_a_flag + '_' + net +\
        '_ec_' + str(ent_coef) + '_lr_' + str(lr) +\
        '_lrs_' + lr_sch + '_g_' + str(gamma) +\
        '_b_' + ut.num2str(nsteps) + '_d_' + ut.num2str(tot_num_stps) +\
        '_ne_' + str(num_env) + '_nu_' + str(nlstm)
    save_path = save_path.replace('-v0', '')
    save_path = save_path.replace('constant', 'c')
    save_path = save_path.replace('linear', 'l')
    # load_path = save_path + '/checkpoints/00020'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    command = 'python -m baselines.run --alg=' + alg + ' --env=' + env +\
        ' --network=' + net + ' --nsteps=' + str(nsteps) +\
        ' --num_timesteps=' + str(tot_num_stps) +\
        ' --log_interval=' + str(li) + ' --ent_coef=' + str(ent_coef) +\
        ' --lrschedule=' + lr_sch + ' --gamma=' + str(gamma) +\
        ' --num_env=' + str(num_env) + tr_h_cmmd +\
        ps_rw_cmmd + ps_a_cmmd + timing_cmmd + ' --lr=' + str(lr) +\
        ' --save_path=' + save_path + ' --nlstm=' + str(nlstm) + \
        load_path_cmmd
    print(command)
    vars_ = vars()
    params = {x: vars_[x] for x in vars_.keys() if type(vars_[x]) == str or
              type(vars_[x]) == int or type(vars_[x]) == float or
              type(vars_[x]) == bool or type(vars_[x]) == list or
              type(vars_[x]) == tuple}
    np.savez(save_path + '/params.npz', ** params)
    os.system(command)

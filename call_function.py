#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:20:00 2019

@author: linux
"""
import os
import utils as ut
import numpy as np
import itertools
import matplotlib
from pathlib import Path
home = str(Path.home())
matplotlib.use('Agg')


def build_command(ps_r=True, ps_act=True, bl_dur=200, num_u=32, stimEv=1.,
                  net_type='twin_net', num_stps_env=1e9, load_path='',
                  save=True):
    alg = 'a2c'
    env = 'RDM-v0'
    nsteps = 20
    num_env = 10
    tot_num_stps = num_stps_env*num_env
    num_steps_per_logging = 1000000
    li = num_steps_per_logging / nsteps
    ent_coef = 0.1
    lr = 1e-3
    lr_sch = 'constant'
    gamma = 0.9
    if net_type == 'twin_net':
        nlstm = num_u // 2
    else:
        nlstm = num_u

    if env == 'RDM-v0':
        timing = [100, 200, 200, 200, 100]
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
        tr_h_flag = '_TH_' + ut.list_str(rep_prob) + '_' + str(bl_dur)
        tr_h_cmmd = ' --trial_hist=True --bl_dur=' + str(bl_dur) +\
            ' --rep_prob '
        for ind_rp in range(len(rep_prob)):
            tr_h_cmmd += str(rep_prob[ind_rp]) + ' '
    else:
        tr_h_flag = ''
        tr_h_cmmd = ''
    # pass reward
    if ps_r:
        ps_rw_flag = '_PR'
        ps_rw_cmmd = ' --pass_reward=True'
    else:
        ps_rw_flag = ''
        ps_rw_cmmd = ''
    # pass action
    if ps_act:
        ps_a_flag = '_PA'
        ps_a_cmmd = ' --pass_action=True'
    else:
        ps_a_flag = ''
        ps_a_cmmd = ''

    load_path = load_path
    if load_path == '':
        load_path_cmmd = ''
    else:
        load_path_cmmd = ' --load_path=' + load_path

    save_path = home + '/neurogym/' + alg + '_' + env + timing_flag + \
        tr_h_flag + ps_rw_flag + ps_a_flag + '_' + net_type
    save_path += '_ec_' + str(ent_coef)
    save_path += '_lr_' + str(lr)
    save_path += '_lrs_' + lr_sch
    save_path += '_g_' + str(gamma)
    save_path += '_b_' + ut.num2str(nsteps)
    save_path += '_d_' + ut.num2str(tot_num_stps)
    save_path += '_ne_' + str(num_env)
    save_path += '_nu_' + str(nlstm)
    save_path += '_ev_' + str(stimEv)
    save_path = save_path.replace('-v0', '')
    save_path = save_path.replace('constant', 'c')
    save_path = save_path.replace('linear', 'l')
    # load_path = save_path + '/checkpoints/00020'
    if not os.path.exists(save_path) and save:
        os.mkdir(save_path)
    command = 'python -m baselines.run --alg=' + alg
    command += ' --env=' + env
    command += ' --network=' + net_type
    command += ' --nsteps=' + str(nsteps)
    command += ' --num_timesteps=' + str(tot_num_stps)
    command += ' --log_interval=' + str(li)
    command += ' --ent_coef=' + str(ent_coef)
    command += ' --lrschedule=' + lr_sch
    command += ' --gamma=' + str(gamma)
    command += ' --num_env=' + str(num_env)
    command += ' --lr=' + str(lr)
    command += ' --save_path=' + save_path
    command += ' --nlstm=' + str(nlstm)
    command += ' --stimEv=' + str(stimEv)
    command += tr_h_cmmd
    command += ps_rw_cmmd
    command += ps_a_cmmd
    command += timing_cmmd
    command += load_path_cmmd
    command += ' --figs=True'
    if save:
        print(command)
        vars_ = vars()
        params = {x: vars_[x] for x in vars_.keys() if type(vars_[x]) == str or
                  type(vars_[x]) == int or type(vars_[x]) == float or
                  type(vars_[x]) == bool or type(vars_[x]) == list or
                  type(vars_[x]) == tuple}
        np.savez(save_path + '/params.npz', ** params)
    else:
        print('Path: ')
        print(save_path)
    return command, save_path


if __name__ == '__main__':
    pass_reward = [True]
    pass_action = [True]
    bl_dur = [200]
    num_units = [64]
    net_type = ['twin_net', 'cont_rnn']  # ['twin_net', 'cont_rnn']
    num_steps = [1e8]  # [1e9]
    stim_ev = [0.5]
    load_path = ''  # '/home/linux/00010'
    params_config = itertools.product(pass_reward, pass_action, bl_dur,
                                      num_units, net_type, num_steps, stim_ev)
    batch_command = ''
    for conf in params_config:
        cmmd, _ = build_command(ps_r=conf[0], ps_act=conf[1], bl_dur=conf[2],
                                num_u=conf[3], net_type=conf[4],
                                num_stps_env=conf[5], load_path=load_path,
                                stimEv=conf[6])
        batch_command += cmmd + '\n'
    os.system(batch_command)

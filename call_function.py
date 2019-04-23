#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:20:00 2019

@author: linux
"""
import os
import numpy as np
import itertools
import matplotlib
from pathlib import Path
from datetime import datetime
from neurogym.ops import utils as ut
home = str(Path.home())
matplotlib.use('Agg')


def build_command(save_folder='/rigel/theory/users/mm5514/',
                  run_folder='/rigel/home/mm5514/',
                  ps_r=True, ps_act=True, bl_dur=200, num_u=32, stimEv=1.,
                  net_type='twin_net', num_stps_env=1e9, load_path='',
                  save=True, nsteps=20, inst=0):
    seed = datetime.now().microsecond
    alg = 'a2c'
    env = 'RDM-v0'
    num_env = 24
    tot_num_stps = num_stps_env*num_env
    num_steps_per_logging = 1000000
    li = num_steps_per_logging / nsteps
    ent_coef = 0.05  # 0.1
    lr = 1e-3
    lr_sch = 'constant'
    gamma = .8  # 0.9
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

    save_path = save_folder + alg + '_' + env +\
        timing_flag + tr_h_flag + ps_rw_flag + ps_a_flag + '_' + net_type
    save_path += '_ec_' + str(ent_coef)
    save_path += '_lr_' + str(lr)
    save_path += '_lrs_' + lr_sch
    save_path += '_g_' + str(gamma)
    save_path += '_b_' + ut.num2str(nsteps)
    save_path += '_d_' + ut.num2str(tot_num_stps)
    save_path += '_ne_' + str(num_env)
    save_path += '_nu_' + str(nlstm)
    save_path += '_ev_' + str(stimEv)
    save_path += '_' + str(seed)
    save_path = save_path.replace('-v0', '')
    save_path = save_path.replace('constant', 'c')
    save_path = save_path.replace('linear', 'l')
    # load_path = save_path + '/checkpoints/00020'
    if not os.path.exists(save_path) and save:
        os.mkdir(save_path)
    command = run_folder + 'run.py --alg=' + alg
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
    command += ' --seed=' + str(seed)
    command += tr_h_cmmd
    command += ps_rw_cmmd
    command += ps_a_cmmd
    command += timing_cmmd
    command += load_path_cmmd
    # command += ' --figs=True'
    print('Path: ')
    print(save_path)
    if save:
        print('Command:')
        print(command)
        vars_ = vars()
        params = {x: vars_[x] for x in vars_.keys() if type(vars_[x]) == str or
                  type(vars_[x]) == int or type(vars_[x]) == float or
                  type(vars_[x]) == bool or type(vars_[x]) == list or
                  type(vars_[x]) == tuple}
        np.savez(save_path + '/params.npz', ** params)

    return command, save_path


def produce_sh_files(cluster='hab'):
    if cluster == 'hab':
        save_folder = '/rigel/theory/users/mm5514/'
        run_folder = '/rigel/home/mm5514/'
    else:
        save_folder = '/gpfs/projects/hcli64/manuel/results/'
        run_folder = '/gpfs/projects/hcli64/manuel/code/'
    home = str(Path.home())
    pass_reward = True
    pass_action = True
    bl_dur = [200]
    num_units = [44]  # [32, 64]
    net_type = ['twin_net']  # ['twin_net', 'cont_rnn']
    num_steps_env = 1e8  # [1e9]
    stim_ev = [.5]  # [.3, .6, 1.]
    batch_size = [20]  # [5, 20]
    insts = np.arange(10)
    load_path = ''  # '/home/linux/00010'
    params_config = itertools.product(insts, bl_dur, num_units, stim_ev,
                                      batch_size, net_type)
    main_file = file = open(home + '/scripts/main_' + cluster + '.sh', 'w')
    command = specs(cluster=cluster)
    main_file.write(command)

    for conf in params_config:
        name = 'scripts/' + str(conf[1]) + '_' + str(conf[2]) +\
            '_' + str(conf[3]) + '_' + str(conf[4]) + '_' +\
            str(conf[5]) + '_' + str(conf[0]) + '_' + cluster + '.sh'
        main_file.write('sbatch ' + name + '\n')
        main_file.write('sleep 10\n')
        file = open(home + '/' + name, 'w')
        cmmd = specs(conf=conf, cluster=cluster)
        aux, _ = build_command(save_folder=save_folder, run_folder=run_folder,
                               inst=conf[0], ps_r=pass_reward,
                               ps_act=pass_action,
                               bl_dur=conf[1], num_u=conf[2],
                               net_type=conf[5], num_stps_env=num_steps_env,
                               load_path=load_path, stimEv=conf[3],
                               nsteps=conf[4], save=False)
        cmmd += aux
        file.write(cmmd)
        file.close()
    main_file.close()


def specs(conf=None, cluster='hab'):
    command = ''
    command += '#!/bin/sh\n'
    if cluster == 'hab':
        command += '#SBATCH --account=theory\n'
        if conf is None:
            command += '#SBATCH --job-name=RUN\n'
            command += '#SBATCH -c 1\n'
            command += '#SBATCH --time=0:30:00\n'
            command += '#SBATCH --mem-per-cpu=128gb\n'
        else:
            name = str(conf[2])
            for ind in range(3, len(conf)):
                name += '_' + str(conf[ind])
            command += '#SBATCH --job-name=' + name + '\n'
            command += '#SBATCH --cpus-per-task=24\n'
            command += '#SBATCH --time=120:00:00\n'
            command += '#SBATCH --mem-per-cpu=5gb\n'
            command += '#SBATCH --exclusive\n'
            command += 'module load anaconda/3-5.1\n'
            command += 'module load tensorflow/anaconda3-5.1.0/1.7.0\n'
    else:
        if conf is None:
            command += '#SBATCH --job-name=RUN\n'
            command += '#SBATCH -c 1\n'
            command += '#SBATCH --time=0:30:00\n'
        else:
            name = str(conf[2])
            for ind in range(3, len(conf)):
                name += '_' + str(conf[ind])
            command += '#SBATCH --job-name=' + name + '\n'
            command += '#SBATCH --cpus-per-task=40\n'
            command += '#SBATCH --time=48:00:00\n'
            command += '#SBATCH --exclusive\n'
            command += 'module purge\n'
            command += 'module load gcc/6.4.0\n'
            command += 'module load cuda/9.1\n'
            command += 'module load cudnn/7.1.3\n'
            command += 'module load openmpi/3.0.0\n'
            command += 'module load atlas/3.10.3\n'
            command += 'module load scalapack/2.0.2\n'
            command += 'module load fftw/3.3.7\n'
            command += 'module load szip/2.1.1\n'
            command += 'module load opencv/3.4.1\n'
            command += 'module load python/3.6.5_ML\n'

    return command


def specs_bsc(conf=None):
    command = ''
    command += '#!/bin/sh\n'
    if conf is None:
        command += '#SBATCH --job-name=RUN\n'
        command += '#SBATCH -c 1\n'
        command += '#SBATCH --time=0:30:00\n'
    else:
        name = str(conf[2])
        for ind in range(3, len(conf)):
            name += '_' + str(conf[ind])
        command += '#SBATCH --job-name=' + name + '\n'
        command += '#SBATCH --cpus-per-task=40\n'
        command += '#SBATCH --time=48:00:00\n'
        command += '#SBATCH --exclusive\n'
        command += 'module load anaconda/3-5.1\n'
        command += 'module load tensorflow/anaconda3-5.1.0/1.7.0\n'
    return command


def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    aadhf = argparse.ArgumentDefaultsHelpFormatter
    return argparse.ArgumentParser(formatter_class=aadhf)


def neuro_arg_parser():
    """
    Create an argparse.ArgumentParser for neuro environments
    """
    parser = arg_parser()
    parser.add_argument('--num_insts',
                        help='number of instances to run',
                        type=int, default=1)
    parser.add_argument('--pass_reward',
                        help='whether to pass the prev. reward with obs',
                        type=bool, default=True)
    parser.add_argument('--pass_action',
                        help='whether to pass the prev. action with obs',
                        type=bool, default=True)
    parser.add_argument('--net_type', help='type of architecture',
                        type=str, default='twin_net')
    parser.add_argument('--num_u',
                        help='number of total units',
                        type=int, nargs='+', default=(32,))
    parser.add_argument('--n_steps',
                        help='rollout',
                        type=float, nargs='+', default=(20,))
    parser.add_argument('--bl_dur',
                        help='dur. of block in the trial-hist wrappr (trials)',
                        type=int, nargs='+', default=(200,))
    parser.add_argument('--stimEv', help='allows scaling stimulus evidence',
                        type=float, nargs='+', default=(1.,))
    return parser


def main(args):
    arg_parser = neuro_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    pass_reward = args.pass_reward
    pass_action = args.pass_action
    bl_dur = args.bl_dur
    num_units = args.num_u
    net_type = args.net_type
    num_steps_env = 1e7  # [1e9]
    stim_ev = args.stimEv
    batch_size = args.n_steps
    insts = np.arange(args.num_insts)
    load_path = ''  # '/home/linux/00010'
    params_config = itertools.product(insts, bl_dur, num_units, stim_ev,
                                      batch_size)
    batch_command = ''
    for conf in params_config:
        print('---------------------------------------------')
        cmmd, _ = build_command(inst=conf[0], ps_r=pass_reward,
                                ps_act=pass_action,
                                bl_dur=conf[1], num_u=conf[2],
                                net_type=net_type, num_stps_env=num_steps_env,
                                load_path=load_path, stimEv=conf[3],
                                nsteps=conf[4])
        batch_command += cmmd + '\n'
    os.system(batch_command)


if __name__ == '__main__':
    produce_sh_files()
    #    asdsad
    #    main(sys.argv)

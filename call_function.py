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
from datetime import datetime
from neurogym.ops import utils as ut
from os.path import expanduser
home = expanduser("~")
matplotlib.use('Agg')


def build_command(save_folder='/rigel/theory/users/mm5514/',
                  run_folder='/rigel/home/mm5514/',
                  ps_r=True, ps_act=True, bl_dur=200, num_u=32, stimEv=1.,
                  net_type='twin_net', num_stps_env=1e9, load_path='',
                  save=True, nsteps=20, alg='a2c', env='RDM-v0',
                  seed=None, seed_task=None, num_env=24, ent_coef=0.05,
                  lr_sch='constant', gamma=.8, rep_prob=(.2, .8),  lr=1e-3,
                  timing=[100, 200, 200, 200, 100], save_folder_name='',
                  eval_steps=100000, alpha=0.1, env2='GNG-v0', delay=[500],
                  timing2=[100, 200, 200, 200, 100, 100], combine=False,
                  trial_hist=False, noise=0, num_steps_per_logging=500000,
                  sv_neural=True, blk_ch_prob=None):
    if seed is None:
        seed = datetime.now().microsecond
    if seed_task is None:
        seed_task = datetime.now().microsecond
    tot_num_stps = num_stps_env*num_env
    li = num_steps_per_logging // nsteps
    if net_type == 'twin_net':
        nlstm = num_u // 2
    else:
        nlstm = num_u
    # duration of different periods
    if len(timing) > 0:
        timing_flag = '_t_' + ut.list_str(timing)
        timing_cmmd = ' --timing '
        for ind_t in range(len(timing)):
            timing_cmmd += str(timing[ind_t]) + ' '
    else:
        timing_flag = ''
        timing_cmmd = ''
    # trial history
    if trial_hist:
        if blk_ch_prob is None:
            tr_h_flag = '_TH_' + ut.list_str(rep_prob) + '_' + str(bl_dur)
            tr_h_cmmd = ' --trial_hist=True --bl_dur=' + str(bl_dur) +\
                ' --rep_prob '
        else:
            tr_h_flag = '_TH_' + ut.list_str(rep_prob) + '_' + str(blk_ch_prob)
            tr_h_cmmd = ' --trial_hist=True' +\
                ' --blk_ch_prob=' + str(blk_ch_prob) + ' --rep_prob '

        for ind_rp in range(len(rep_prob)):
            tr_h_cmmd += str(rep_prob[ind_rp]) + ' '
    else:
        tr_h_flag = ''
        tr_h_cmmd = ''
    # combine
    if combine:
        comb_flag = '_COMB_' + ut.list_str(rep_prob) + '_env2_' + env2 +\
            '_' + str(delay) + '_t2_' + ut.list_str(timing2)
        comb_cmmd = ' --combine=True --delay=' + str(delay) +\
            ' --env2=' + env2 + ' --timing2 '
        for ind_t in range(len(timing2)):
            comb_cmmd += str(timing2[ind_t]) + ' '
    else:
        comb_flag = ''
        comb_cmmd = ''
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
    if sv_neural:
        sv_n_cmmd = ' --save_neural_data=True'
    else:
        sv_n_cmmd = ''
    save_path = save_folder + alg + '_' + env +\
        timing_flag + comb_flag + tr_h_flag + ps_rw_flag + ps_a_flag +\
        '_' + net_type
    save_path += '_ec_' + str(ent_coef)
    save_path += '_lr_' + str(lr)
    save_path += '_lrs_' + lr_sch
    save_path += '_g_' + str(gamma)
    save_path += '_b_' + ut.num2str(nsteps)
    save_path += '_ne_' + str(num_env)
    save_path += '_nu_' + str(nlstm)
    save_path += '_ev_' + str(stimEv)
    save_path += '_a_' + str(alpha)
    save_path += '_n_' + str(noise)
    save_path += '_' + str(seed)
    save_path += str(seed_task)
    save_path += save_folder_name
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
    command += ' --seed_task=' + str(seed_task)
    command += ' --eval_steps=' + str(eval_steps)
    command += ' --alpha=' + str(alpha)
    command += ' --sigma_rec=' + str(noise)
    command += tr_h_cmmd
    command += comb_cmmd
    command += ps_rw_cmmd
    command += ps_a_cmmd
    command += timing_cmmd
    command += load_path_cmmd
    command += sv_n_cmmd
    # command += ' --figs=True'
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


def produce_sh_files(cluster='hab', alg=['supervised'], hours='120',
                     num_units=[32], bl_dur=[200], stim_ev=[.5],
                     rep_prob=[[.2, .8]], batch_size=[20],
                     net_type=['cont_rnn'], pass_r=[True], pass_act=[True],
                     num_insts=5, experiment='', main_folder='',
                     num_steps_env=1e8, alpha=[0.1],
                     combine=False, tr_hist=False, noise=[0], env='RDM-v0',
                     env2='GNG-v0', delay=[500],
                     timing=[100, 200, 200, 200, 100],
                     timing2=[100, 200, 200, 200, 100, 100],
                     scripts_folder='', seed=None, seed_task=None,
                     sv_neural=True, blk_ch_probs=[None]):
    if cluster == 'hab':
        save_folder = main_folder + experiment + '/'
        run_folder = '/rigel/home/mm5514/'
        n_envs = 24
    else:
        save_folder = main_folder + experiment + '/'
        run_folder = '/home/hcli64/hcli64348/'
        n_envs = 40
    insts = np.arange(num_insts)
    params_config = itertools.product(batch_size, bl_dur, num_units, stim_ev,
                                      net_type, pass_r, pass_act, alg,
                                      rep_prob, alpha, delay, noise,
                                      blk_ch_probs, insts)
    scr_folder = scripts_folder + experiment + '/'
    if not os.path.exists(scr_folder):
        os.makedirs(scr_folder)
    main_file = open(scr_folder + 'main_' +
                     cluster + '.sh', 'w')

    for conf in params_config:
        print('-----------------------')
        name = env[:-3] + '_'
        if tr_hist:
            name += ut.list_str(conf[8])
        if combine:
            name += str(conf[10])
        for ind in np.arange(8):
            name += '_' + str(conf[ind])
        for ind in [9, 11, 12, 13]:
            name += '_' + str(conf[ind])
        name += '_' + hours + 'h_' + cluster
        name = name.replace('.', '')
        name += '.sh'
        print(name)
        main_file.write('sbatch ' + name + '\n')
        main_file.write('sleep 20\n')
        # training script
        file = open(scr_folder + name, 'w')
        cmmd = specs(conf=conf, cluster=cluster, hours=hours, alg=conf[7],
                     name=name, n_envs=n_envs)
        aux, _ = build_command(save_folder=save_folder, run_folder=run_folder,
                               ps_r=conf[5], ps_act=conf[6], rep_prob=conf[8],
                               bl_dur=conf[1], num_u=conf[2], num_env=n_envs,
                               net_type=conf[4], num_stps_env=num_steps_env,
                               load_path='', stimEv=conf[3], noise=conf[11],
                               nsteps=conf[0], save=False, alg=conf[7],
                               eval_steps=0, alpha=conf[9], delay=conf[10],
                               timing=timing, timing2=timing2,
                               env=env, env2=env2, combine=combine,
                               trial_hist=tr_hist,
                               seed=seed, seed_task=seed_task,
                               sv_neural=sv_neural, blk_ch_prob=conf[12])
        cmmd += aux
        file.write(cmmd)
        file.close()

    main_file.close()
    analysis_file = open(scr_folder + 'analysis_' +
                         cluster + '.sh', 'w')
    command = specs(cluster=cluster, hours='2')
    command += run_folder + 'analysis.py ' + save_folder
    analysis_file.write(command)
    analysis_file.close()


def specs(conf=None, cluster='hab', hours='120', alg='a2c', name='',
          n_envs=24):
    command = ''
    command += '#!/bin/sh\n'
    if cluster == 'hab':
        command += '#SBATCH --account=theory\n'
        if conf is None:
            command += '#SBATCH --job-name=RUN\n'
            command += '#SBATCH -c 1\n'
            command += '#SBATCH --time=' + hours + ':00:00\n'
            command += '#SBATCH --mem-per-cpu=128gb\n'
            command += 'module load anaconda/3-5.1\n'
            command += 'module load tensorflow/anaconda3-5.1.0/1.7.0\n'
        else:
            name = name[:-3] + '_' + hours
            command += '#SBATCH --job-name=' + name + '\n'
            command += '#SBATCH --cpus-per-task=' + str(n_envs) + '\n'
            command += '#SBATCH --time=' + hours + ':00:00\n'
            command += '#SBATCH --mem-per-cpu=5gb\n'
            command += '#SBATCH --exclusive\n'
            command += 'module load anaconda/3-5.1\n'
            command += 'module load tensorflow/anaconda3-5.1.0/1.7.0\n'
    else:
        if conf is None:
            command += '#SBATCH --job-name=RUN\n'
            command += '#SBATCH --output=RUN.sh\n'
            command += '#SBATCH -c 1\n'
            command += '#SBATCH --time=' + hours + ':00:00\n'
            command += 'module purge\n'
            command += 'module load gcc/6.4.0\n'
            command += 'module load cuda/9.1\n'
            command += 'module load cudnn/7.1.3\n'
            command += 'module load openmpi/3.0.0\n'
            command += 'module load atlas/3.10.3\n'
            command += 'module load scalapack/2.0.2\n'
            command += 'module load fftw/3.3.7\n'
            command += 'module load szip/2.1.1\n'
            command += 'module load ffmpeg\n'
            command += 'module load opencv/3.4.1\n'
            command += 'module load python/3.6.5_ML\n'
        else:
            name = name[:-3]
            command += '#SBATCH --job-name=' + name + '\n'
            command += '#SBATCH --output=' + name + '.out\n'
            command += '#SBATCH --cpus-per-task=' + str(n_envs) + '\n'
            command += '#SBATCH --time=' + hours + ':00:00\n'
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
            command += 'module load ffmpeg\n'
            command += 'module load opencv/3.4.1\n'
            command += 'module load python/3.6.5_ML\n'

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
        cmmd, _ = build_command(ps_r=pass_reward,
                                ps_act=pass_action,
                                bl_dur=conf[1], num_u=conf[2],
                                net_type=net_type, num_stps_env=num_steps_env,
                                load_path=load_path, stimEv=conf[3],
                                nsteps=conf[4])
        batch_command += cmmd + '\n'
    os.system(batch_command)


if __name__ == '__main__':
    # A2C ALGORITHM
    hours = '4'
    alg = ['a2c']
    num_units = [64]
    batch_size = [20]
    net_type = ['cont_rnn']
    pass_r = [True]
    pass_act = [True]
    num_insts = 10
    num_steps_env = 1e8

    cluster = 'bsc'  # 'hab'
    project = 'neurogym'
    # main_folder = '/rigel/theory/users/mm5514/'
    main_folder = '/gpfs/projects/hcli64/molano/neurogym/'
    scripts_folder = home + '/' + project + '/' + cluster + '_scripts/'
    if not os.path.exists(scripts_folder):
        os.makedirs(scripts_folder)
    all_scripts_file = open(scripts_folder + 'all_scripts.sh', 'w')
    command = ''
    envs_list = ['Mante-v0',
                 'Romo-v0',
                 'RDM-v0',
                 'padoaSch-v0',
                 'pdWager-v0',
                 'DPA-v0',
                 'GNG-v0',
                 'ReadySetGo-v0',
                 'DelayedMatchSample-v0',
                 'DawTwoStep-v0',
                 'MatchingPenny-v0',
                 'Bandit-v0']
    timing_list = [(200, 300, 200, 300, 400, 200),
                   (100, 200, 200, 200, 200, 200),
                   (200, 200, 300, 400, 200),
                   (200, 200, 400, 200),
                   (200, 300, 400, 500, 200, 300, 400, 200, 200, 200, 200),
                   (100, 200, 200, 400, 200, 100, 200),
                   (100, 200, 200, 200, 100, 100),
                   (500, 100, 100),
                   (200, 200, 500, 200, 200),
                   (),
                   (),
                   ()]
    for ind, env in enumerate(envs_list):
        experiment = env[:-3]
        print(timing_list[ind])
        produce_sh_files(env=env, cluster=cluster, alg=alg, hours=hours,
                         num_units=num_units,
                         batch_size=batch_size, net_type=net_type,
                         pass_r=pass_r, pass_act=pass_act,
                         num_insts=num_insts, experiment=experiment,
                         main_folder=main_folder, num_steps_env=num_steps_env,
                         timing=timing_list[ind],
                         scripts_folder=scripts_folder)
        command += 'cd ' + experiment + '\n'
        command += 'bash ' + 'main_' + cluster + '.sh\n'
        command += 'cd ..\n'

    # DUAL TASK
    combine = True
    delay = [500]
    timing = [100, 200, 600, 600, 200, 100, 100]
    timing2 = [100, 200, 200, 200, 100, 100]

    experiment = 'dual_task'

    produce_sh_files(env='DPA-v0', env2='GNG-v0', cluster=cluster, alg=alg,
                     hours=hours,
                     num_units=num_units,
                     batch_size=batch_size, net_type=net_type,
                     pass_r=pass_r, pass_act=pass_act,
                     num_insts=num_insts, experiment=experiment,
                     main_folder=main_folder, num_steps_env=num_steps_env,
                     combine=combine, delay=delay, timing=timing,
                     timing2=timing2,
                     scripts_folder=scripts_folder)
    command += 'cd ' + experiment + '\n'
    command += 'bash ' + 'main_' + cluster + '.sh\n'
    command += 'cd ..\n'

    # PRIORS
    timing = (200, 200, 300, 400, 200)
    bl_dur = [200]
    rep_prob = [[.2, .8]]
    stim_ev = [.5]
    experiment = 'priors'
    tr_hist = True
    produce_sh_files(cluster=cluster, alg=alg, hours=hours, tr_hist=tr_hist,
                     num_units=num_units,
                     bl_dur=bl_dur, stim_ev=stim_ev, rep_prob=rep_prob,
                     batch_size=batch_size, net_type=net_type,
                     pass_r=pass_r, pass_act=pass_act, timing=timing,
                     num_insts=num_insts, experiment=experiment,
                     main_folder=main_folder, num_steps_env=num_steps_env,
                     scripts_folder=scripts_folder)
    command += 'cd ' + experiment + '\n'
    command += 'bash ' + 'main_' + cluster + '.sh\n'
    command += 'cd ..\n'

    # SUPERVISED LEARNING
    alg = ['supervised']
    envs_list = ['Mante-v0',
                 'Romo-v0',
                 'RDM-v0',
                 'DPA-v0',
                 'GNG-v0',
                 'ReadySetGo-v0',
                 'DelayedMatchSample-v0',
                 ]
    timing_list = [(200, 300, 200, 300, 400, 200),
                   (100, 200, 200, 200, 200, 200),
                   (200, 200, 300, 400, 200),
                   (100, 200, 200, 400, 200, 100, 200),
                   (100, 200, 200, 200, 100, 100),
                   (500, 100, 100),
                   (200, 200, 500, 200, 200),
                   ]
    for ind, env in enumerate(envs_list):
        experiment = env[:-3] + '_supervised'

        produce_sh_files(env=env, cluster=cluster, alg=alg, hours=hours,
                         num_units=num_units,
                         batch_size=batch_size, net_type=net_type,
                         pass_r=pass_r, pass_act=pass_act,
                         num_insts=num_insts, experiment=experiment,
                         main_folder=main_folder, num_steps_env=num_steps_env,
                         timing=timing_list[ind],
                         scripts_folder=scripts_folder)
        command += 'cd ' + experiment + '\n'
        command += 'bash ' + 'main_' + cluster + '.sh\n'
        command += 'cd ..\n'

    # DUAL TASK
    combine = True
    delay = [500]
    timing = [100, 200, 600, 600, 200, 100, 100]
    timing2 = [100, 200, 200, 200, 100, 100]
    experiment = 'dual_task_supervised'

    produce_sh_files(env='DPA-v0', env2='GNG-v0', cluster=cluster, alg=alg,
                     hours=hours,
                     num_units=num_units,
                     batch_size=batch_size, net_type=net_type,
                     pass_r=pass_r, pass_act=pass_act,
                     num_insts=num_insts, experiment=experiment,
                     main_folder=main_folder, num_steps_env=num_steps_env,
                     combine=combine, delay=delay, timing=timing,
                     timing2=timing2,
                     scripts_folder=scripts_folder)
    command += 'cd ' + experiment + '\n'
    command += 'bash ' + 'main_' + cluster + '.sh\n'
    command += 'cd ..\n'

    # PRIORS
    timing = (200, 200, 300, 400, 200)
    bl_dur = [200]
    rep_prob = [[.2, .8]]
    stim_ev = [.5]
    experiment = 'priors_supervised'
    tr_hist = True
    produce_sh_files(cluster=cluster, alg=alg, hours=hours, tr_hist=tr_hist,
                     num_units=num_units,
                     bl_dur=bl_dur, stim_ev=stim_ev, rep_prob=rep_prob,
                     batch_size=batch_size, net_type=net_type,
                     pass_r=pass_r, pass_act=pass_act,
                     num_insts=num_insts, experiment=experiment,
                     main_folder=main_folder, num_steps_env=num_steps_env,
                     scripts_folder=scripts_folder)
    command += 'cd ' + experiment + '\n'
    command += 'bash ' + 'main_' + cluster + '.sh\n'
    command += 'cd ..\n'
    all_scripts_file.write(command)
    all_scripts_file.close()

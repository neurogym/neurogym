#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:56:14 2019

@author: linux
"""
import numpy as np
import glob
import os
from pathlib import Path
import sys
import json
home = str(Path.home())
sys.path.append(home + '/neurogym')
from neurogym.ops import put_together_files as ptf
from neurogym import call_function as cf
non_relevant_params = {'seed': 0, 'save_path': 0, 'log_interval': 0,
                       'num_timesteps': 0, 'eval_steps': 0}


def load(file='/home/linux/params.npz'):
    params = np.load(file)
    args = vars(params['args'].tolist())
    n_args = vars(params['n_args'].tolist())
    extra_args = params['extra_args'].tolist()
    args.update(n_args)
    args.update(extra_args)
    return args


def compare_dicts(x, y):
    assert len(x) == len(y)
    non_shared_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
    different = False
    for param in non_shared_items.keys():
        if param not in non_relevant_params.keys():
            different = True
            break
    if different:
        return False, non_shared_items
    else:
        return True, []


def check_new_exp(experiments, args, params_explored):
    new = True
    for ind_exps in range(len(experiments)):
        same_exp, non_shared = compare_dicts(experiments[ind_exps][0], args)
        if same_exp:
            experiments[ind_exps].append(args)
            new = False
            group = ind_exps
            break
        params_explored.update(non_shared)
    if new:
        experiments.append([args])
        group = len(experiments) - 1

    return experiments, params_explored, group


def write_script(conf, save_folder, run_folder, load_folder, args, train_more):
    retr_ev_scr = run_folder+'/re_training_evaluating_scripts/'
    if not os.path.exists(retr_ev_scr):
        os.mkdir(retr_ev_scr)
    if train_more:
        num_steps = args['num_timesteps']
        sv_f_name = '/post_training/'
        file_name = 'ptrain.sh'
        eval_steps = 0
    else:
        num_steps = 0
        sv_f_name = '/evaluating/'
        file_name = 'ev.sh'
        eval_steps = 100000

    cmmd = cf.specs(conf=conf, cluster='hab', hours='4', alg=args['alg'])
    aux, _ = cf.build_command(save_folder=save_folder,
                              run_folder=run_folder,
                              seed=args['seed'],
                              ps_r=args['pass_reward'],
                              ps_act=args['pass_action'],
                              bl_dur=args['bl_dur'],
                              num_u=args['nlstm'],
                              net_type=args['network'],
                              num_stps_env=num_steps,
                              load_path=load_folder,
                              stimEv=args['stimEv'],
                              nsteps=args['nsteps'], save=False,
                              alg=args['alg'],
                              timing=args['timing'],
                              num_env=args['num_env'],
                              ent_coef=args['ent_coef'], lr=args['lr'],
                              lr_sch=args['lrschedule'],
                              gamma=args['gamma'],
                              rep_prob=args['rep_prob'],
                              save_folder_name=sv_f_name,
                              eval_steps=eval_steps)
    name_aux = os.path.basename(os.path.normpath(load_folder + '/'))
    with open(retr_ev_scr + name_aux + '_' + file_name, 'w') as file:
        cmmd += aux
        file.write(cmmd)
        file.close()


def explore_folder(main_folder, count=True,
                   save_folder='/rigel/theory/users/mm5514/',
                   run_folder='/rigel/home/mm5514/'):
    params_explored = {}
    experiments = []
    num_trials = []
    folders = glob.glob(main_folder + '/*')
    for ind_f in range(len(folders)):
        path = folders[ind_f]
        folder = os.path.basename(os.path.normpath(path + '/'))
        file = path + '/params.npz'
        if os.path.exists(file):
            args = load(file)
            args = update_exp(args, folder, file, main_folder, key='alpha',
                              value=0.1, look_for='_a_', replace_with='_a_0.1')

            if len(experiments) == 0:
                experiments.append([args])
                group = 0
            else:
                experiments, params_explored, group =\
                    check_new_exp(experiments, args, params_explored)
            # count number of trials
            if count:
                flag = ptf.put_files_together(path, min_num_trials=1)
                if flag:
                    data = np.load(path + '/bhvr_data_all.npz')
                    num_tr = data['choice'].shape[0]
                else:
                    num_tr = 0
                if len(num_trials) == 0:
                    num_trials.append([num_tr])
                elif group > len(num_trials)-1:
                    num_trials.append([num_tr])
                else:
                    num_trials[group].append(num_tr)

    params_explored = {k: args[k] for k in params_explored
                       if k not in non_relevant_params}

    args = experiments[0][0]
    p_exp = {k: args[k] for k in args if k not in params_explored}
    main_file = open('results.sh', 'w')
    main_file.write('common params\n')
    main_file.write(json.dumps(p_exp))
    main_file.write('\nxxxxxxxxxxxxxxxx\n')
    for ind_exps in range(len(experiments)):
        args = experiments[ind_exps][0]
        p_exp = {k: args[k] for k in args if k in params_explored}
        main_file.write(json.dumps(p_exp))
        main_file.write('\nnumber of instances: ' +
                        str(len(experiments[ind_exps])) + '\n')
        if count:
            main_file.write('number of trials per instance:' +
                            str(num_trials[ind_exps]) + '\n')
        main_file.write('------------------------\n')
    main_file.close()
    data = {'experiments': experiments}
    np.savez(main_folder + '/experiments.npz', **data)
    return experiments, params_explored


def update_exp(args, folder, file, main_folder, key='alpha', value=0.1,
               look_for='_a_', replace_with='_a_0.1'):
    if key not in args.keys():
        args[key] = value
    np.savez(file, **args)
    if folder.find(look_for) == -1:
        aux = folder.rfind('_')
        new_name = folder.replace(folder[aux:], replace_with+folder[aux:])
        os.rename(main_folder+'/'+folder, main_folder+'/'+new_name)
    return args


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = home + '/mm5514/'
    explore_folder(main_folder)

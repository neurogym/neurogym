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
home = str(Path.home())
sys.path.append(home + '/neurogym')
from neurogym.ops import put_together_files as ptf


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
    shared_items = {k: x[k] for k in x if k in y and x[k] == y[k]}
    non_shared_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
    if len(shared_items) == len(x)-1 and 'seed' in non_shared_items.keys():
        return True, []
    else:
        shared_items = {k: x[k] for k in x if k in y and x[k] != y[k]}
        return False, non_shared_items
 

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


def explore_folder(main_folder):
    params_explored = {}
    experiments = []
    num_trials = []
    folders = glob.glob(main_folder + '/*')
    for ind_f in range(len(folders)):
        file = folders[ind_f] + '/params.npz'
        if os.path.exists(file):
            args = load(file)
            if len(experiments) == 0:
                experiments.append([args])
                group = 0
            else:
                experiments, params_explored, group =\
                    check_new_exp(experiments, args, params_explored)
            # count number of trials
            flag = ptf.put_files_together(folders[ind_f], min_num_trials=1)
            if flag:
                data = np.load(folders[ind_f] + '/bhvr_data_all.npz')
                num_tr= data['choice'].shape[0]
            else:
                num_tr = 0
            if len(num_trials) == 0:
                num_trials.append([num_tr])
            elif group > len(num_trials)-1:
                num_trials.append([num_tr])
            else:
                num_trials[group].append(num_tr)
    args = experiments[0][0]
    p_exp = {k: args[k] for k in args if k not in params_explored}
    print('common params')
    print(p_exp)
    print('xxxxxxxxxxxxxxxx')
    for ind_exps in range(len(experiments)):
        args = experiments[ind_exps][0]
        p_exp = {k: args[k] for k in args if k in params_explored}
        print(p_exp)
        print('number of instances: ' + str(len(experiments[ind_exps])))
        print('number of trials per instance:' + str(num_trials[ind_exps]))
        print('------------------------')
        
        
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = home + '/mm5514/'
    explore_folder(main_folder)
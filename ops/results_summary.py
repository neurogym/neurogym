#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:56:14 2019

@author: linux
"""
import numpy as np


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
    if len(shared_items) == len(args)-1 and 'seed' in non_shared_items.keys():
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
            break
        params_explored.update(non_shared)
    if new:
        experiments.append([args])
    return experiments, params_explored

    
if __name__ == '__main__':
    params_explored = {}
    experiments = []
    args = load()
    experiments.append([args])
    args2 = args.copy()
    args2['seed'] = 12355
    experiments, params_explored =\
        check_new_exp(experiments, args2, params_explored)
    args3 = args.copy()
    args3['env'] = 'asd'
    experiments, params_explored =\
        check_new_exp(experiments, args3, params_explored)
    print(experiments)
    print(params_explored)
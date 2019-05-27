#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:20:01 2019

@author: linux
"""
# ['choice',
# 'stimulus',
# 'correct_side',
# 'obs_mat',
# 'act_mat',
# 'rew_mat',
# 'rep_prob']
import glob
import numpy as np
from neurogym.ops import utils as ut
# import os


def put_files_together(folder, min_num_trials=1e6):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    files = ut.order_by_sufix(files)
    choice_mat = np.empty(())
    stim_mat = np.empty(())
    side_mat = np.empty(())
    SIZE = 0
    for ind_f in range(len(files)):
        loaded = True
        try:
            data = np.load(files[ind_f])
        except:
            loaded = False
        if loaded:
            choice = data['choice']
            stim = data['stimulus']
            side = data['correct_side']
            side = side.reshape((side.shape[0], -1))
            if side.shape[1] != 1:
                side = np.argmax(side, axis=1)
            else:
                side = side.reshape((side.shape[0],))
            choice_mat = np.concatenate((choice_mat, choice))
            stim_mat = np.concatenate((stim_mat, stim))
            side_mat = np.concatenate((side_mat, side))
            SIZE += choice.shape[0]
    if SIZE > 0:
        print(np.array(choice_mat).shape)
        data = {'choice': choice_mat, 'stimulus': stim_mat,
                'correct_side': side_mat}
        np.savez(folder + '/bhvr_data_all.npz', **data)
        if SIZE > min_num_trials:
            return True
        else:
            return False

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


def put_together_files(folder, min_num_trials=1e6):
    files = glob.glob(folder + '/*_bhvr_data*npz')
    files = ut.order_by_sufix(files)
    choice_mat = np.empty((0,))
    side_mat = np.empty((0,))
    reward_mat = np.empty((0,))
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
            if 'reward' in data.keys():
                reward = data['reward']
            else:
                reward = np.array([])
            side = side.reshape((side.shape[0], -1))
            if side.shape[1] != 1:
                side = np.argmax(side, axis=1)
            else:
                side = side.reshape((side.shape[0],))
            choice_mat = np.concatenate((choice_mat, choice))
            reward_mat = np.concatenate((reward_mat, reward))
            if SIZE == 0:  # define stim_mat using stim shape
                stim_mat = np.empty((0, stim.shape[1]))
            stim_mat = np.concatenate((stim_mat, stim))
            side_mat = np.concatenate((side_mat, side))
            SIZE += choice.shape[0]
    if SIZE > 0:
        data = {'choice': choice_mat, 'stimulus': stim_mat,
                'correct_side': side_mat, 'reward': reward_mat}
        np.savez(folder + '/bhvr_data_all.npz', **data)
        if SIZE >= min_num_trials:
            return True
        else:
            print('not enough data (' + str(SIZE) + ')')
            return False
    else:
        print('not enough data (0)')

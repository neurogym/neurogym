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
    choice_mat = []
    stim_mat = []
    r_prob_mat = []
    side_mat = []
    counter = 0
    SIZE = 0
    for ind_f in range(len(files)):
        loaded = True
        try:
            data = np.load(files[ind_f])
            counter += 1
        except:
            loaded = False
        if loaded:
            choice = data['choice']
            stim = data['stimulus']
            r_prob = data['rep_prob']
            side = data['correct_side']
            side = side.reshape((side.shape[0], -1))
            if side.shape[1] != 1:
                side = np.argmax(side, axis=1)
            else:
                side = side.reshape((side.shape[0],))
            if choice.shape[0] != side.shape[0]:
                dec_time = np.where(stim[:, 0] == 0)[0]
                dec_time_aux = np.concatenate((dec_time,
                                               np.array([dec_time[-1]+2])))
                dec_time_aux = np.diff(dec_time_aux)
                assert (dec_time_aux >= 1).all()
                dec_time = dec_time[dec_time_aux != 1]
                choice = choice[dec_time]
                stim = stim[dec_time-1, :]
            choice_mat.append(choice)
            stim_mat.append(stim)
            r_prob_mat.append(r_prob)
            side_mat.append(side)
            SIZE += choice.shape[0]
#            if ind_f == 0:
#                SIZE = choice.shape
#            assert (SIZE == choice.shape), str(SIZE) + ' ' + str(choice.shape)
#            assert (SIZE[0] ==
#                    stim.shape[0]), str(SIZE) + ' ' + str(stim.shape)
#            rps = r_prob.shape
#            assert (SIZE[0] == rps[0]), str(SIZE) + ' ' + str(rps)
#            assert (SIZE == side.shape), str(SIZE) + ' ' + str(side.shape)
    if counter > 0:
        choice_mat = np.reshape(np.array(choice_mat), (SIZE, ))
        stim_mat = np.reshape(np.array(stim_mat), (SIZE,
                                                   stim.shape[1]))
        side_mat = np.reshape(np.array(side_mat), (SIZE, ))
        # r_prob_mat = np.reshape(np.array(r_prob_mat), (SIZE[0]*counter,))
        r_prob_mat = []
        data = {'choice': choice_mat, 'stimulus': stim_mat,
                'correct_side': side_mat, 'rep_prob': r_prob_mat}
        np.savez(folder + '/bhvr_data_all.npz', **data)
        if SIZE*counter > min_num_trials:
            return True
        else:
            return False
